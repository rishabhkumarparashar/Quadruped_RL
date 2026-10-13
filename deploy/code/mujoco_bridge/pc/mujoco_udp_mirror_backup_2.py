#!/usr/bin/env python3
"""Live Grallator real-hardware -> MuJoCo joint mirror.

Input packet: the existing JSON telemetry emitted by the patched
main_controller.py on the Jetson.  For each named joint, qf is the calibrated
real encoder position and vf is the calibrated encoder velocity.

Modes:
  direct : networking/joint-map validation. q_real is copied directly into
           MuJoCo qpos and no physics is integrated.
  pd     : q_real becomes q_des for MuJoCo. The XML torque motors are driven
           by tau = Kp*(q_des-q_sim) + Kd*(qd_des-qd_sim), then physics steps.
           Use this mode when you are ready to compare dynamics / add PACE.
"""

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "BR_hip_joint", "BR_thigh_joint", "BR_calf_joint",
    "BL_hip_joint", "BL_thigh_joint", "BL_calf_joint",
]


class LatestJointPacket:
    def __init__(self, bind_host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_host, port))
        self._sock.settimeout(0.2)
        self._lock = threading.Lock()
        self._q = {}
        self._qd = {}
        self._last_rx = None
        self._source = None
        self._count = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass

    def snapshot(self):
        with self._lock:
            return (
                dict(self._q),
                dict(self._qd),
                self._last_rx,
                self._source,
                self._count,
            )

    def _run(self):
        while not self._stop:
            try:
                raw, source = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                packet = json.loads(raw.decode("utf-8"))
                q = {}
                qd = {}

                # Current standalone Jetson encoder packet:
                # {"timestamp": ..., "q": {"FR_hip_joint": ..., ...}}
                if isinstance(packet.get("q"), dict):
                    for name, value in packet["q"].items():
                        if name in JOINTS:
                            q[name] = float(value)

                # Also keep compatibility with older telemetry format.
                else:
                    for item in packet.get("joints", []):
                        name = item.get("n")
                        if name not in JOINTS:
                            continue
                        if item.get("qf") is not None:
                            q[name] = float(item["qf"])
                        if item.get("vf") is not None:
                            qd[name] = float(item["vf"])

                if not q:
                    continue
            except (ValueError, TypeError, json.JSONDecodeError):
                continue

            with self._lock:
                self._q = q
                self._qd = qd
                self._last_rx = time.monotonic()
                self._source = source
                self._count += 1


def name_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return idx


def reset_to_keyframe(model, data, keyframe="ZERO_STAND"):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def build_map(model):
    result = {}
    for name in JOINTS:
        jid = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor")
        result[name] = {
            "jid": jid,
            "qpos": int(model.jnt_qposadr[jid]),
            "dof": int(model.jnt_dofadr[jid]),
            "act": aid,
            "lo": float(model.jnt_range[jid, 0]),
            "hi": float(model.jnt_range[jid, 1]),
        }
    return result


def apply_direct(model, data, mapping, q_real, qd_real):
    for name, m in mapping.items():
        if name not in q_real:
            continue
        data.qpos[m["qpos"]] = float(q_real[name])
        data.qvel[m["dof"]] = float(qd_real.get(name, 0.0))
    mujoco.mj_forward(model, data)


def apply_pd(model, data, mapping, q_des, qd_des, kp, kd, torque_limit):
    data.ctrl[:] = 0.0
    for name, m in mapping.items():
        if name not in q_des:
            continue
        target = float(np.clip(q_des[name], m["lo"], m["hi"]))
        target_vel = float(qd_des.get(name, 0.0))
        q = float(data.qpos[m["qpos"]])
        qd = float(data.qvel[m["dof"]])
        tau = kp * (target - q) + kd * (target_vel - qd)
        data.ctrl[m["act"]] = np.clip(tau, -torque_limit, torque_limit)


# ============================================================
# REAL-TIME FOOT END-TO-END MEASUREMENTS
# ============================================================

FOOT_COLLISION_GEOMS = {
    "FL": "FL_foot_collision",
    "FR": "FR_foot_collision",
    "BL": "BL_foot_collision",
    "BR": "BR_foot_collision",
}


def build_foot_geom_map(model):
    result = {}

    for leg, geom_name in FOOT_COLLISION_GEOMS.items():

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if gid < 0:
            raise RuntimeError(
                f"Foot collision geom not found: {geom_name}"
            )

        radius = float(
            model.geom_size[gid, 0]
        )

        result[leg] = {
            "id": gid,
            "name": geom_name,
            "radius": radius,
        }

    return result


def get_foot_centers(data, foot_map):

    return {
        leg: np.array(
            data.geom_xpos[info["id"]],
            dtype=float,
        )
        for leg, info in foot_map.items()
    }


def end_to_end_measurements_mm(data, foot_map):

    p = get_foot_centers(
        data,
        foot_map,
    )

    r = {
        leg: foot_map[leg]["radius"]
        for leg in foot_map
    }

    # --------------------------------------------------------
    # Front pair: FL outside edge -> FR outside edge
    # --------------------------------------------------------

    front_width = (
        abs(
            p["FL"][1]
            - p["FR"][1]
        )
        + r["FL"]
        + r["FR"]
    ) * 100.0


    # --------------------------------------------------------
    # Rear pair: BL outside edge -> BR outside edge
    # --------------------------------------------------------

    rear_width = (
        abs(
            p["BL"][1]
            - p["BR"][1]
        )
        + r["BL"]
        + r["BR"]
    ) * 100.0


    # --------------------------------------------------------
    # Left side: FL front edge -> BL rear edge
    # --------------------------------------------------------

    left_length = (
        abs(
            p["FL"][0]
            - p["BL"][0]
        )
        + r["FL"]
        + r["BL"]
    ) * 100.0


    # --------------------------------------------------------
    # Right side: FR front edge -> BR rear edge
    # --------------------------------------------------------

    right_length = (
        abs(
            p["FR"][0]
            - p["BR"][0]
        )
        + r["FR"]
        + r["BR"]
    ) * 100.0


    # --------------------------------------------------------
    # Overall outermost foot envelope
    # --------------------------------------------------------

    leftmost = min(
        p[leg][1] - r[leg]
        for leg in p
    )

    rightmost = max(
        p[leg][1] + r[leg]
        for leg in p
    )

    rearmost = min(
        p[leg][0] - r[leg]
        for leg in p
    )

    frontmost = max(
        p[leg][0] + r[leg]
        for leg in p
    )

    overall_width = (
        rightmost - leftmost
    ) * 100.0

    overall_length = (
        frontmost - rearmost
    ) * 100.0


    return {
        "front_width": front_width,
        "rear_width": rear_width,
        "left_length": left_length,
        "right_length": right_length,
        "overall_width": overall_width,
        "overall_length": overall_length,
    }


def print_end_to_end_measurements(model, data, foot_map):

    d = end_to_end_measurements_mm(
        data,
        foot_map,
    )

    print(
        f"FOOT END-TO-END [cm] | "
        f"FRONT={d['front_width']:7.1f} | "
        f"REAR={d['rear_width']:7.1f} | "
        f"LEFT={d['left_length']:7.1f} | "
        f"RIGHT={d['right_length']:7.1f} | "
        f"OVERALL W={d['overall_width']:7.1f} | "
        f"OVERALL L={d['overall_length']:7.1f}"
    )




# ============================================================
# FOOT POINT VISUALIZATION
# ============================================================

def add_marker_geom(viewer, pos, rgba, radius=0.010):
    """
    Draw one small sphere marker in the viewer user scene.
    """
    scn = viewer.user_scn

    if scn.ngeom >= scn.maxgeom:
        return

    g = scn.geoms[scn.ngeom]

    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius], dtype=float),
        pos=np.array(pos, dtype=float),
        mat=np.eye(3, dtype=float).reshape(-1),
        rgba=np.array(rgba, dtype=float),
    )

    scn.ngeom += 1


def get_foot_edge_points(model, data, foot_map):
    """
    Put one GREEN marker at the DISTAL TIP of each foot.

    The tip is defined as the point on the spherical foot collision
    geom that lies directly away from that leg's calf.

        direction = foot_center - calf_center
        tip       = foot_center + radius * unit(direction)

    This makes the marker move and rotate naturally with the leg.
    """

    result = {}

    calf_geom_names = {
        "FL": "FL_calf_collision",
        "FR": "FR_calf_collision",
        "BL": "BL_calf_collision",
        "BR": "BR_calf_collision",
    }

    for leg, info in foot_map.items():

        foot_gid = info["id"]
        r = float(info["radius"])

        foot_center = np.array(
            data.geom_xpos[foot_gid],
            dtype=float,
        )

        calf_gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            calf_geom_names[leg],
        )

        if calf_gid < 0:
            raise RuntimeError(
                f"Could not find {calf_geom_names[leg]}"
            )

        calf_center = np.array(
            data.geom_xpos[calf_gid],
            dtype=float,
        )

        direction = foot_center - calf_center

        norm = float(np.linalg.norm(direction))

        if norm < 1e-9:
            raise RuntimeError(
                f"{leg}: calf-to-foot direction is zero"
            )

        direction /= norm

        tip_world = (
            foot_center
            + r * direction
        )

        result[leg] = {
            "front": tip_world,
        }

    return result


def draw_foot_edge_markers(viewer, model, data, foot_map, marker_radius):
    """
    Draw:

      - one GREEN marker at the front point of each foot
      - very thin GREEN measurement lines
      - distance labels in centimeters at the line midpoints

    Measurements:
      FRONT = FL green <-> FR green
      BACK  = BL green <-> BR green
      LEFT  = FL green <-> BL green
      RIGHT = FR green <-> BR green

    Distance is horizontal XY distance between marker centers.
    """

    scn = viewer.user_scn

    # Clear our previous custom visualization geoms every frame.
    scn.ngeom = 0

    points = get_foot_edge_points(
        model,
        data,
        foot_map,
    )

    # --------------------------------------------------------
    # GREEN POINTS
    # --------------------------------------------------------

    green_points = {}

    for leg in ("FL", "FR", "BL", "BR"):

        p_green = np.array(
            points[leg]["front"],
            dtype=float,
        )

        green_points[leg] = p_green

        add_marker_geom(
            viewer,
            p_green,
            rgba=[0.0, 1.0, 0.0, 1.0],
            radius=marker_radius,
        )


    # --------------------------------------------------------
    # HELPER: THIN LINE
    # --------------------------------------------------------

    def add_measurement_line(a, b):
        if scn.ngeom >= scn.maxgeom:
            return

        g = scn.geoms[scn.ngeom]

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3, dtype=float),
            pos=np.zeros(3, dtype=float),
            mat=np.eye(3, dtype=float).reshape(-1),
            rgba=np.array(
                [0.0, 1.0, 0.0, 0.9],
                dtype=float,
            ),
        )

        # Width is in pixels.
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            1.0,
            np.asarray(a, dtype=float),
            np.asarray(b, dtype=float),
        )

        scn.ngeom += 1


    # --------------------------------------------------------
    # HELPER: 3D TEXT LABEL
    # --------------------------------------------------------

    def add_distance_label(position, text_label):
        if scn.ngeom >= scn.maxgeom:
            return

        g = scn.geoms[scn.ngeom]

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.zeros(3, dtype=float),
            pos=np.asarray(position, dtype=float),
            mat=np.eye(3, dtype=float).reshape(-1),
            rgba=np.array(
                [1.0, 1.0, 1.0, 1.0],
                dtype=float,
            ),
        )

        # MuJoCo's visualization geom contains a text-label field.
        try:
            g.label = text_label
        except TypeError:
            g.label = text_label.encode("utf-8")

        scn.ngeom += 1


    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    def distance_cm(a, b):
        """
        Horizontal XY distance between GREEN marker centers.
        """
        delta = (
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )

        return float(
            np.linalg.norm(delta[:2]) * 100.0
        )


    # --------------------------------------------------------
    # THE FOUR MEASUREMENT LINES
    # --------------------------------------------------------

    pairs = [
        (
            "FRONT",
            green_points["FL"],
            green_points["FR"],
        ),

        (
            "BACK",
            green_points["BL"],
            green_points["BR"],
        ),

        (
            "LEFT",
            green_points["FL"],
            green_points["BL"],
        ),

        (
            "RIGHT",
            green_points["FR"],
            green_points["BR"],
        ),
    ]


    for label, a, b in pairs:

        # Draw line exactly between the two GREEN centers.
        add_measurement_line(
            a,
            b,
        )

        d_cm = distance_cm(
            a,
            b,
        )

        # Put text at midpoint of the line.
        midpoint = (
            np.asarray(a, dtype=float)
            + np.asarray(b, dtype=float)
        ) * 0.5

        # Lift text 3 cm vertically so it does not sit inside the line.
        midpoint[2] += 0.03

        add_distance_label(
            midpoint,
            f"{label}: {d_cm:.2f} cm",
        )


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=here / "model" / "Grallator_Final.xml")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=57543)
    parser.add_argument("--mode", choices=("direct", "pd"), default="direct")
    parser.add_argument("--kp", type=float, default=250.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--torque-limit", type=float, default=100.0)
    parser.add_argument("--stale-ms", type=float, default=250.0)

    parser.add_argument(
        "--foot-measure-hz",
        type=float,
        default=0.0,
        help="Rate for printing live foot end-to-end dimensions.",
    )

    parser.add_argument(
        "--show-foot-points",
        action="store_true",
        help="Draw front/rear/left/right colored markers on each foot.",
    )
    parser.add_argument(
        "--foot-point-radius",
        type=float,
        default=0.010,
        help="Marker sphere radius in meters for foot point highlighting.",
    )

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data)
    mapping = build_map(model)

    foot_map = build_foot_geom_map(model)

    print()
    print("Foot collision geometry:")

    for leg in ("FL", "FR", "BL", "BR"):
        info = foot_map[leg]

        print(
            f"  {leg}: "
            f"{info['name']}  "
            f"radius={info['radius'] * 100.0:.1f} cm"
        )


    print()
    print("Foot point colors:")
    print("  GREEN = front point only")

    receiver = LatestJointPacket(args.bind, args.port)
    receiver.start()

    print("=" * 72)
    print("GRALLATOR REAL -> MUJOCO UDP MIRROR")
    print("=" * 72)
    print(f"XML       : {args.xml}")
    print(f"UDP bind  : {args.bind}:{args.port}")
    print(f"Mode      : {args.mode}")
    print(f"MuJoCo dt : {model.opt.timestep:.6f} s")
    if args.mode == "direct":
        print("Behavior  : calibrated real encoder q is copied directly to MuJoCo qpos")
    else:
        print(f"Behavior  : real encoder q -> MuJoCo q_des, PD Kp={args.kp}, Kd={args.kd}")
        print(f"Tau limit : +/-{args.torque_limit:.1f} Nm")
    print("Waiting for Jetson packets...\n")

    sim_dt = float(model.opt.timestep)
    last_print = 0.0
    last_packet_count = -1

    last_foot_measure_print = 0.0

    foot_measure_period = (
        1.0 / args.foot_measure_hz
        if args.foot_measure_hz > 0.0
        else float("inf")
    )



    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_start = time.perf_counter()
                q_real, qd_real, last_rx, source, packet_count = receiver.snapshot()
                now = time.monotonic()
                fresh = last_rx is not None and (now - last_rx) <= args.stale_ms / 1000.0

                if args.mode == "direct":
                    if q_real:
                        apply_direct(model, data, mapping, q_real, qd_real)
                    time.sleep(0.005)
                else:
                    if fresh:
                        apply_pd(
                            model, data, mapping, q_real, qd_real,
                            args.kp, args.kd, args.torque_limit,
                        )
                    else:
                        # Never continue applying a stale remote target indefinitely.
                        data.ctrl[:] = 0.0

                    mujoco.mj_step(model, data)
                    elapsed = time.perf_counter() - loop_start
                    if elapsed < sim_dt:
                        time.sleep(sim_dt - elapsed)

                if args.show_foot_points:
                    draw_foot_edge_markers(
                        viewer,
                        model,
                        data,
                        foot_map,
                        args.foot_point_radius,
                    )

                viewer.sync()

                if (
                    q_real
                    and now - last_foot_measure_print >= foot_measure_period
                ):
                    print_end_to_end_measurements(
                        model,
                        data,
                        foot_map,
                    )

                    last_foot_measure_print = now



                if packet_count != last_packet_count and now - last_print >= 1.0:
                    last_packet_count = packet_count
                    last_print = now
                    age_ms = None if last_rx is None else 1000.0 * (now - last_rx)
                    source_text = "none" if source is None else f"{source[0]}:{source[1]}"
                    age_text = "n/a" if age_ms is None else f"{age_ms:.1f} ms"
                    print(
                        f"packets={packet_count:7d}  source={source_text:21s} "
                        f"age={age_text:>10s}  joints={len(q_real):2d}"
                    )
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
