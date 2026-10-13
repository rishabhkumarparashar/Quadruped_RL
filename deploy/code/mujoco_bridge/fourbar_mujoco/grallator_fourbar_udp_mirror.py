#!/usr/bin/env python3
"""
Live Grallator four-bar hardware -> MuJoCo mirror.

Purpose
-------
Read one real motor encoder angle from a UDP JSON stream, treat that motor
as the four-bar crank, solve the exact four-bar geometry, and place the
MuJoCo crank/coupler/rocker joints at the corresponding closed-loop pose.

This is READ-ONLY with respect to the robot:
    real encoder -> UDP -> this script -> MuJoCo
It never sends a motor command.

Expected packet example
-----------------------
{
    "q": {
        "FR_calf_joint": 0.523599
    }
}

By default q is assumed to be in radians.

IMPORTANT CALIBRATION
---------------------
--geom-angle-at-start is the physical crank angle, measured from ground link
AD to crank AB, at the instant the first valid UDP packet is received.

For example, if the physical crank is 60 deg above AD at startup:
    --geom-angle-at-start 60

The first real encoder sample becomes the encoder reference, so only the
encoder DELTA is added to that known physical geometric reference.

This avoids incorrectly assuming that motor encoder zero == geometric 0 deg.
"""

import argparse
import json
import math
import socket
import time
from pathlib import Path

import mujoco
import mujoco.viewer


# -----------------------------
# Exact Grallator four-bar data
# -----------------------------
AB = 0.05530
BC = 0.34367
CD = 0.05530
AD = 0.34000

# XML reference configuration.
XML_THETA2_REF = math.radians(60.0)


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def solve_fourbar(theta2, branch=+1):
    """
    Solve planar four-bar geometry.

    theta2:
        Global crank AB angle measured from +AD direction.

    Returns:
        theta3 = global coupler BC angle
        theta4 = global rocker DC angle

    branch:
        +1 upper/open intersection
        -1 lower/crossed intersection
    """
    bx = AB * math.cos(theta2)
    by = AB * math.sin(theta2)

    dx, dy = AD, 0.0
    vx = dx - bx
    vy = dy - by
    L = math.hypot(vx, vy)

    if L < 1e-12:
        return None

    if L > BC + CD + 1e-10:
        return None
    if L < abs(BC - CD) - 1e-10:
        return None

    x = (BC * BC - CD * CD + L * L) / (2.0 * L)
    h2 = BC * BC - x * x
    if h2 < -1e-10:
        return None
    h = math.sqrt(max(0.0, h2))

    ux = vx / L
    uy = vy / L

    px = bx + x * ux
    py = by + x * uy

    cx = px + branch * h * (-uy)
    cy = py + branch * h * ux

    theta3 = math.atan2(cy - by, cx - bx)
    theta4 = math.atan2(cy - dy, cx - dx)

    return theta3, theta4


XML_SOL = solve_fourbar(XML_THETA2_REF, +1)
if XML_SOL is None:
    raise RuntimeError("Internal XML reference geometry is invalid.")

XML_THETA3_REF, XML_THETA4_REF = XML_SOL
XML_COUPLER_REL_REF = wrap_pi(XML_THETA3_REF - XML_THETA2_REF)


def get_joint_qpos_adr(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"Joint not found in XML: {joint_name}")
    return int(model.jnt_qposadr[jid])


def geometric_pose_to_qpos(theta2, theta3, theta4):
    """
    Convert global link angles into the three MuJoCo hinge qpos values.

    XML body reference orientations already contain the 60-degree valid pose,
    so qpos values are offsets from that reference pose.
    """
    q_crank = wrap_pi(theta2 - XML_THETA2_REF)

    current_coupler_relative = wrap_pi(theta3 - theta2)
    q_coupler = wrap_pi(current_coupler_relative - XML_COUPLER_REL_REF)

    q_rocker = wrap_pi(theta4 - XML_THETA4_REF)

    return q_crank, q_coupler, q_rocker


def parse_args():
    here = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Mirror a real Grallator motor encoder into a MuJoCo four-bar."
    )
    p.add_argument(
        "--xml",
        default=str(here / "grallator_fourbar.xml"),
        help="Path to grallator_fourbar.xml",
    )
    p.add_argument(
        "--bind-ip",
        default="0.0.0.0",
        help="UDP bind IP on the PC",
    )
    p.add_argument(
        "--port",
        type=int,
        default=5005,
        help="UDP port carrying encoder JSON packets",
    )
    p.add_argument(
        "--joint",
        default="FR_calf_joint",
        help="Key inside packet['q'] to use as the real crank encoder",
    )
    p.add_argument(
        "--encoder-units",
        choices=("rad", "deg"),
        default="rad",
        help="Units used by the incoming encoder value",
    )
    p.add_argument(
        "--encoder-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Use -1 only if physical crank rotation is opposite the chosen geometric sign",
    )
    p.add_argument(
        "--geom-angle-at-start",
        type=float,
        default=64.309702,
        help=(
            "Physical crank angle AB relative to ground AD, in degrees, "
            "at the first received encoder sample"
        ),
    )
    p.add_argument(
        "--branch",
        type=int,
        choices=(-1, 1),
        default=1,
        help="+1=open/upper branch, -1=crossed/lower branch",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Viewer/update loop rate in Hz",
    )
    p.add_argument(
        "--print-rate",
        type=float,
        default=10.0,
        help="Terminal status print rate in Hz",
    )
    return p.parse_args()



# ============================================================
# LIVE_THIGH_CALF_ANGLE_OVERLAY
# Draw live thigh-calf angle directly inside MuJoCo viewer.
#
# thigh direction = A -> D
# calf direction  = opposite D -> C_rocker
# ============================================================

def draw_thigh_calf_angle(viewer, data, site_a, site_b, site_d, site_c):
    """
    Draw CAD-style live thigh-calf angle.

    DISPLAY:
        arc is drawn INSIDE the physical thigh/calf wedge:
            D -> A      = thigh ray
            D -> calf   = calf ray

    VALUE:
        reports the mechanical thigh-calf angle used previously,
        i.e. the supplementary angle.
    """

    A = [
        float(data.site_xpos[site_a][0]),
        float(data.site_xpos[site_a][1]),
        float(data.site_xpos[site_a][2]),
    ]

    B = [
        float(data.site_xpos[site_b][0]),
        float(data.site_xpos[site_b][1]),
        float(data.site_xpos[site_b][2]),
    ]

    D = [
        float(data.site_xpos[site_d][0]),
        float(data.site_xpos[site_d][1]),
        float(data.site_xpos[site_d][2]),
    ]

    C = [
        float(data.site_xpos[site_c][0]),
        float(data.site_xpos[site_c][1]),
        float(data.site_xpos[site_c][2]),
    ]

    # ========================================================
    # LIVE ANGLE VALUE
    # ========================================================

    # D -> A = physical thigh ray from knee
    thigh_axis = [
        A[0] - D[0],
        A[1] - D[1],
    ]

    # D -> calf = opposite of D -> C rocker
    calf_axis = [
        D[0] - C[0],
        D[1] - C[1],
    ]

    lt = math.hypot(thigh_axis[0], thigh_axis[1])
    lc = math.hypot(calf_axis[0], calf_axis[1])

    if lt < 1e-12 or lc < 1e-12:
        return float("nan")

    thigh_axis[0] /= lt
    thigh_axis[1] /= lt

    calf_axis[0] /= lc
    calf_axis[1] /= lc

    thigh_axis_angle = math.atan2(
        thigh_axis[1], thigh_axis[0]
    )

    calf_angle = math.atan2(
        calf_axis[1], calf_axis[0]
    )

    mechanical_sweep = wrap_pi(
        calf_angle - thigh_axis_angle
    )

    angle_deg = 180.0 - abs(math.degrees(mechanical_sweep))

    # ========================================================
    # DRAWING DIRECTIONS
    #
    # For the CAD-style picture we draw between the ACTUAL
    # physical links leaving the knee:
    #
    #       D -> A      thigh
    #       D -> calf   calf
    #
    # This puts the arc INSIDE the mechanism.
    # ========================================================

    thigh_draw_angle = math.atan2(
        A[1] - D[1],
        A[0] - D[0],
    )

    calf_draw_angle = calf_angle

    draw_sweep = wrap_pi(
        calf_draw_angle - thigh_draw_angle
    )

    # ========================================================
    # CUSTOM VIEWER GEOMETRY
    # ========================================================

    scn = viewer.user_scn
    scn.ngeom = 0

    IDENTITY = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    CYAN = [0.35, 0.82, 1.00, 1.00]
    CYAN_SOFT = [0.45, 0.86, 1.00, 0.90]

    z = D[2] + 0.022

    # IMPORTANT:
    # Angle center is the REAL knee pivot D.
    center = [
        D[0],
        D[1],
        z,
    ]

    RADIUS = 0.085

    def add_line(p0, p1, width=2.0, rgba=CYAN):
        i = scn.ngeom
        g = scn.geoms[i]

        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            rgba,
        )

        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            p0,
            p1,
        )

        scn.ngeom += 1

    # ========================================================
    # GUIDE LINES
    # ========================================================

    start_angle = thigh_draw_angle
    end_angle = thigh_draw_angle + draw_sweep

    guide_radius = RADIUS * 1.18

    thigh_guide = [
        center[0] + guide_radius * math.cos(start_angle),
        center[1] + guide_radius * math.sin(start_angle),
        z,
    ]

    calf_guide = [
        center[0] + guide_radius * math.cos(end_angle),
        center[1] + guide_radius * math.sin(end_angle),
        z,
    ]

    add_line(center, thigh_guide, 1.5, CYAN_SOFT)
    add_line(center, calf_guide, 1.5, CYAN_SOFT)

    # ========================================================
    # ARC
    # ========================================================

    N = 64

    previous = [
        center[0] + RADIUS * math.cos(start_angle),
        center[1] + RADIUS * math.sin(start_angle),
        z,
    ]

    arc_start = previous[:]

    for k in range(1, N + 1):

        t = k / N

        a = start_angle + draw_sweep * t

        current = [
            center[0] + RADIUS * math.cos(a),
            center[1] + RADIUS * math.sin(a),
            z,
        ]

        add_line(previous, current, 2.2, CYAN)

        previous = current

    arc_end = previous[:]

    # ========================================================
    # ARROWHEADS
    # ========================================================

    direction_sign = 1.0 if draw_sweep >= 0.0 else -1.0

    def add_arrow(tip, tangent, reverse=False):

        tx, ty = tangent

        if reverse:
            tx = -tx
            ty = -ty

        arrow_length = 0.012
        arrow_width = 0.005

        bx = tip[0] - arrow_length * tx
        by = tip[1] - arrow_length * ty

        nx = -ty
        ny = tx

        p1 = [
            bx + arrow_width * nx,
            by + arrow_width * ny,
            z,
        ]

        p2 = [
            bx - arrow_width * nx,
            by - arrow_width * ny,
            z,
        ]

        add_line(tip, p1, 2.0, CYAN)
        add_line(tip, p2, 2.0, CYAN)

    start_tangent = [
        direction_sign * -math.sin(start_angle),
        direction_sign *  math.cos(start_angle),
    ]

    end_tangent = [
        direction_sign * -math.sin(end_angle),
        direction_sign *  math.cos(end_angle),
    ]

    add_arrow(
        arc_start,
        start_tangent,
        reverse=True,
    )

    add_arrow(
        arc_end,
        end_tangent,
        reverse=False,
    )

    # ========================================================
    # LIVE ANGLE TEXT
    # Put it in the middle of the wedge.
    # ========================================================

    middle_angle = (
        start_angle + 0.5 * draw_sweep
    )

    text_radius = RADIUS * 0.58

    label_pos = [
        center[0] + text_radius * math.cos(middle_angle),
        center[1] + text_radius * math.sin(middle_angle),
        z + 0.006,
    ]

    i = scn.ngeom
    g = scn.geoms[i]

    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_LABEL,
        [0.0, 0.0, 0.0],
        label_pos,
        IDENTITY,
        CYAN,
    )

    g.label = f"{angle_deg:.3f} deg"

    scn.ngeom += 1


    # ========================================================
    # FOUR-BAR POINT LABELS
    #
    # A = motor/crank pivot
    # B = crank/coupler joint
    # C = coupler/rocker joint
    # D = fixed knee/rocker pivot
    # E = visible calf endpoint
    # ========================================================

    # E lies on the physical calf ray, opposite D -> C.
    ex = D[0] - C[0]
    ey = D[1] - C[1]
    el = math.hypot(ex, ey)

    if el > 1e-12:
        ex /= el
        ey /= el

        # Match the approximate visible calf extension.
        E = [
            D[0] + 0.240 * ex,
            D[1] + 0.240 * ey,
            D[2],
        ]
    else:
        E = D[:]

    def add_point_label(text, P, dx=0.0, dy=0.0):
        i = scn.ngeom
        g = scn.geoms[i]

        label_pos = [
            P[0] + dx,
            P[1] + dy,
            P[2] + 0.040,
        ]

        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_LABEL,
            [0.0, 0.0, 0.0],
            label_pos,
            IDENTITY,
            [1.0, 0.85, 0.20, 1.0],
        )

        g.label = text
        scn.ngeom += 1

    add_point_label("A", A, -0.015,  0.015)
    add_point_label("B", B,  0.000,  0.018)
    add_point_label("C", C,  0.015,  0.015)
    add_point_label("D", D,  0.015, -0.015)
    add_point_label("E", E,  0.015, -0.010)

    return angle_deg


def main():
    args = parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    qadr_crank = get_joint_qpos_adr(model, "FL_calf_joint")
    qadr_coupler = get_joint_qpos_adr(model, "FL_calf_fourbar_coupler_joint")
    qadr_rocker = get_joint_qpos_adr(model, "FL_calf_fourbar_rocker_joint")

    # Sites used only for the live viewer angle annotation.
    site_a = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "A_marker"
    )

    site_b = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "B_marker"
    )
    site_d = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "D_marker"
    )
    site_c = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "C_rocker"
    )

    if min(site_a, site_b, site_d, site_c) < 0:
        raise RuntimeError(
            "Angle overlay needs sites A_marker, D_marker and C_rocker in the XML."
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    sock.setblocking(False)

    theta2_start = math.radians(args.geom_angle_at_start)
    start_solution = solve_fourbar(theta2_start, args.branch)

    if start_solution is None:
        raise SystemExit(
            f"ERROR: --geom-angle-at-start {args.geom_angle_at_start:.3f} deg "
            "does not close this four-bar."
        )

    theta3_start, theta4_start = start_solution

    latest_encoder = None
    latest_sender = None
    bad_packets = 0

    period = 1.0 / max(args.rate, 1.0)
    print_period = 1.0 / max(args.print_rate, 0.1)
    next_print = 0.0

    print("\nGRALLATOR FOUR-BAR HARDWARE MIRROR")
    print("----------------------------------")
    print(f"AB = {AB*1000:.2f} mm")
    print(f"BC = {BC*1000:.2f} mm")
    print(f"CD = {CD*1000:.2f} mm")
    print(f"AD = {AD*1000:.2f} mm")
    print(f"UDP = {args.bind_ip}:{args.port}")
    print(f"Joint = {args.joint}")
    print(f"Geometric crank angle at first packet = {args.geom_angle_at_start:.3f} deg")
    print("Mode = READ-ONLY hardware -> MuJoCo")
    print("----------------------------------\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Put the model at the requested geometric start pose before data arrives.
        q0 = geometric_pose_to_qpos(theta2_start, theta3_start, theta4_start)
        data.qpos[qadr_crank] = q0[0]
        data.qpos[qadr_coupler] = q0[1]
        data.qpos[qadr_rocker] = q0[2]
        mujoco.mj_forward(model, data)

        # Draw live thigh-calf angle before first UDP packet arrives.
        draw_thigh_calf_angle(
            viewer, data, site_a, site_b, site_d, site_c
        )

        viewer.sync()

        while viewer.is_running():
            loop_start = time.perf_counter()

            # Drain the UDP socket and keep only the newest valid packet.
            while True:
                try:
                    raw, sender = sock.recvfrom(65535)
                except BlockingIOError:
                    break

                try:
                    packet = json.loads(raw.decode("utf-8"))
                    q_dict = packet.get("q", {})
                    if args.joint not in q_dict:
                        bad_packets += 1
                        continue

                    value = float(q_dict[args.joint])
                    if args.encoder_units == "deg":
                        value = math.radians(value)

                    latest_encoder = value
                    latest_sender = sender
                except Exception:
                    bad_packets += 1

            if latest_encoder is not None:
                encoder_delta = args.encoder_sign * latest_encoder
                theta2 = theta2_start + encoder_delta

                sol = solve_fourbar(theta2, args.branch)

                if sol is not None:
                    theta3, theta4 = sol
                    q = geometric_pose_to_qpos(theta2, theta3, theta4)

                    # Exact kinematic placement: no motor command is sent anywhere.
                    data.qpos[qadr_crank] = q[0]
                    data.qpos[qadr_coupler] = q[1]
                    data.qpos[qadr_rocker] = q[2]
                    data.qvel[:] = 0.0

                    mujoco.mj_forward(model, data)

                    crank_delta_deg = math.degrees(theta2 - theta2_start)
                    rocker_delta_deg = math.degrees(wrap_pi(theta4 - theta4_start))

                    if abs(crank_delta_deg) > 1e-9:
                        avg_ratio = rocker_delta_deg / crank_delta_deg
                    else:
                        avg_ratio = float("nan")

                    now = time.perf_counter()
                    if now >= next_print:
                        next_print = now + print_period
                        ratio_text = (
                            f"{avg_ratio:+.6f}"
                            if math.isfinite(avg_ratio)
                            else "reference"
                        )
                        sender_text = (
                            f"{latest_sender[0]}:{latest_sender[1]}"
                            if latest_sender else "-"
                        )

                        print(
                            "\r"
                            f"sender={sender_text:<21} "
                            f"encoder={math.degrees(latest_encoder):+9.3f} deg | "
                            f"dCrank={crank_delta_deg:+8.3f} deg | "
                            f"rocker={math.degrees(theta4):+9.3f} deg | "
                            f"dRocker={rocker_delta_deg:+8.3f} deg | "
                            f"ratio={ratio_text:<10} | "
                            f"bad={bad_packets}",
                            end="",
                            flush=True,
                        )
                else:
                    now = time.perf_counter()
                    if now >= next_print:
                        next_print = now + print_period
                        print(
                            "\r"
                            f"INVALID FOUR-BAR POSE: geometric crank "
                            f"{math.degrees(theta2):+.3f} deg cannot close.     ",
                            end="",
                            flush=True,
                        )

            # Rebuild the arc and live degree label using the
            # CURRENT four-bar pose.
            draw_thigh_calf_angle(
                viewer, data, site_a, site_b, site_d, site_c
            )

            viewer.sync()

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
