#!/usr/bin/env python3
"""Bidirectional Grallator joint viewer, calf four-bar IK and guarded hardware control.

Default: mirror 12 hardware joints into MuJoCo and show live GUI readback.
A companion Tk control window lists all 12 joints and puts MOTOR ENABLE at its top.
The built-in passive MuJoCo viewer does not expose custom interactive checkboxes.

STAGE 1 SAFETY: Only one selected joint may be physically commanded at a time,
matching the existing Jetson mujoco_command_receiver.py v1 backend. Select the
same joint on both ends. The other 11 sliders show actual encoder feedback.
No hardware commands are emitted unless --enable-hardware-command is supplied;
the GUI motor enable also starts OFF and requires fresh Jetson telemetry.

For the four calf joints, GUI sliders and MuJoCo qpos use rocker/calf angles;
hardware UDP q/q_des use calibrated motor/crank-A angles. The conversion is the
four-bar linkage, NOT an assumed 1:1 gear ratio.

Hardware control requires the real suspended robot, local E-stop, and the
Jetson-side motor watchdog/torque/thermal/joint-limit protections. Do not run
another motor-control or STOP/POLL process on the same CAN interfaces.

Original implementation context:

Input packet: the existing JSON telemetry emitted by the patched
main_controller.py on the Jetson.  For each named joint, qf is the calibrated
real encoder position and vf is the calibrated encoder velocity.

Four-bar conversion (default ON):
  - Incoming *_calf_joint q is the physical motor encoder at crank A, in radians.
  - Confirmed motor convention: POSITIVE encoder q rotates AB CLOCKWISE
    when viewed from the CAD drawing side. In the thigh-local CCW-positive
    geometric frame, theta2 = theta2_zero - q_motor.
  - The four-bar rocker at D drives the calf link; convert to calf-joint radians
    before using direct or PD mode. Hip/thigh inputs remain unchanged.
  - The virtual calf coordinate is CLOCKWISE-positive relative to the CAD zero:
    q_calf = theta4_zero - theta4. This matches the motor encoder sign over
    the specified motor range, -14.85 to +83.15 degrees.
  - Incoming calf vf (when present) is converted using dq_calf/dq_motor.
  - Calf sim zero is assumed to coincide with the measured motor-zero pose.
    The MuJoCo joint's positive rotation direction still needs visual checking.
  - Disable for comparison with --no-fourbar.

Modes:
  direct : networking/joint-map validation. q_real is copied directly into
           MuJoCo qpos and no physics is integrated.
  pd     : q_real becomes q_des for MuJoCo. The XML torque motors are driven
           by tau = Kp*(q_des-q_sim) + Kd*(qd_des-qd_sim), then physics steps.
           Use this mode when you are ready to compare dynamics / add PACE.
"""

import argparse
import json
import math
import socket
import sys
import uuid
import tkinter as tk
from tkinter import ttk
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


# ============================================================
# REAL CALF MOTOR AT A -> VIRTUAL CALF JOINT AT D
# ============================================================
# Local (thigh-fixed) 2D geometric frame: A=(0, 0), D=(AD, 0),
# CCW-positive geometric angles, but CLOCKWISE-positive motor encoder q_A
# and CLOCKWISE-positive virtual calf encoder q_D.
# All lengths in mm; all angles entering/leaving the Python functions
# are radians. Encoder readings are relative to the motor-zero pose.
FOURBAR_AB_MM = 55.30  # crank
FOURBAR_BC_MM = 343.67  # coupler
FOURBAR_CD_MM = 55.30  # rocker
FOURBAR_AD_MM = 340.00  # fixed, thigh-mounted base
CRANK_ZERO_DEG = 134.29  # CAD pose at motor encoder q=0
CAD_ROCKER_ZERO_DEG = 129.21  # approximate CAD check, not used as exact zero

CALF_JOINTS = (
    "FL_calf_joint", "FR_calf_joint", "BL_calf_joint", "BR_calf_joint"
)

# HARDWARE motor encoder limits (degrees) supplied on 2026-10-09.
# These are not the same as XML/virtual calf rocker angle limits.
CALF_MOTOR_LIMITS_DEG = {
    "FL_calf_joint": (-14.92, +83.19),
    "BL_calf_joint": ( -6.20, +79.43),
    "FR_calf_joint": (-82.05,  +8.92),
    "BR_calf_joint": (-84.91,  +6.55),
}




def wrap_pi(angle: float) -> float:
    """Represent an angular difference in [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def solve_fourbar_rocker(theta2: float) -> tuple[float, float]:
    """Return (absolute theta4 in rad, dtheta4/dtheta2) for the CAD branch.

    Four-bar closure is solved using the cosine rule on triangles ABD and BCD.
    For the shown Grallator assembly the rocker direction DC is clockwise
    from DB, hence theta4 = beta - alpha with both absolute directions from
    thigh-local +X. The other assembly branch must not be used inadvertently.
    """
    a, b, c, d = (
        FOURBAR_AB_MM, FOURBAR_BC_MM, FOURBAR_CD_MM, FOURBAR_AD_MM
    )
    bx, by = a * math.cos(theta2), a * math.sin(theta2)
    bd = math.hypot(bx - d, by)
    if bd <= 1e-10 or bd < abs(b - c) - 1e-8 or bd > b + c + 1e-8:
        raise ValueError(
            f"four-bar cannot close: crank={math.degrees(theta2):.3f} deg, "
            f"BD={bd:.3f} mm"
        )
    cosine_alpha = (bd * bd + c * c - b * b) / (2.0 * bd * c)
    alpha = math.acos(max(-1.0, min(1.0, cosine_alpha)))
    beta = math.atan2(by, bx - d)  # directed angle of D->B from +X
    theta4 = beta - alpha         # Grallator CAD assembly branch

    # Derive theta3 from solved geometry, then compute exact velocity ratio.
    cx, cy = d + c * math.cos(theta4), c * math.sin(theta4)
    theta3 = math.atan2(cy - by, cx - bx)
    denominator = math.sin(theta3 - theta4)
    if abs(denominator) < 1e-8:
        ratio = float("nan")  # toggle/singular pose: derivative not defined
    else:
        ratio = a * math.sin(theta3 - theta2) / (c * denominator)
    return theta4, ratio


CRANK_ZERO_RAD = math.radians(CRANK_ZERO_DEG)
# Calculate the model's exact zero, rather than subtracting rounded CAD 129.21°.
ROCKER_ZERO_RAD, _ = solve_fourbar_rocker(CRANK_ZERO_RAD)


def calf_motor_to_joint(q_motor: float, qd_motor: float | None = None):
    """Motor encoder at A -> rocker/calf displacement at D (radians).

    Confirmed hardware: increasing q_motor rotates crank AB clockwise.
    We define the virtual calf position q_D clockwise-positive as well,
    with both encoders zero at the same CAD calibration pose.

      theta2 = theta2_zero - q_motor
      q_D = wrap(theta4_zero - theta4(theta2))

    As both input and output reverse their geometric CCW convention,
    dq_D/dq_motor = dtheta4/dtheta2 (the four-bar angular Jacobian).

    Assumes MuJoCo calf qpos=0 is the CAD zero pose, and its positive
    joint direction matches the chosen virtual calf encoder convention.
    """
    if not math.isfinite(q_motor):
        raise ValueError("nonfinite calf motor angle")
    theta2 = CRANK_ZERO_RAD - q_motor
    theta4, ratio = solve_fourbar_rocker(theta2)
    q_calf = wrap_pi(ROCKER_ZERO_RAD - theta4)
    if qd_motor is None:
        return q_calf, None, ratio
    if not math.isfinite(qd_motor) or not math.isfinite(ratio):
        raise ValueError("nonfinite calf motor velocity or singular transmission")
    return q_calf, ratio * qd_motor, ratio



def mirrored_right_motor_to_joint(q_motor: float, qd_motor: float | None = None):
    """FR/BR: mirrored geometry, A=(AD,0), D=(0,0)."""

    if not math.isfinite(q_motor):
        raise ValueError("nonfinite right calf motor angle")

    # Motor A is on the RIGHT; rocker D is on the LEFT.
    # Positive encoder rotates the right crank clockwise.
    theta2_right = math.pi - CRANK_ZERO_RAD - q_motor

    # Reflect geometry into the existing four-bar solver.
    theta2_reflected = math.pi - theta2_right
    theta4_reflected, jacobian = solve_fourbar_rocker(theta2_reflected)

    # Convert rocker orientation back to right-side geometry.
    theta4_right = math.pi - theta4_reflected
    theta4_right_zero = math.pi - ROCKER_ZERO_RAD

    # Right virtual calf joint position
    q_calf = wrap_pi(theta4_right_zero - theta4_right)

    # Signed velocity transmission
    ratio = jacobian

    if qd_motor is None:
        return q_calf, None, ratio

    if not math.isfinite(qd_motor) or not math.isfinite(ratio):
        raise ValueError("invalid right calf velocity")

    return q_calf, ratio * qd_motor, ratio


def convert_calf_packet(q_raw: dict, qd_raw: dict):
    """Convert only the four calf channels; keep hip/thigh channels intact.

    On an invalid four-bar pose, omit that calf target rather than falling
    back to treating the motor encoder as the simulated calf joint angle.
    Returns copied dictionaries, per-calf ratios and a list of warnings.
    """
    q_out, qd_out = dict(q_raw), dict(qd_raw)
    ratios, warnings = {}, []
    for name in CALF_JOINTS:
        if name not in q_raw:
            continue
        try:
            convert = (mirrored_right_motor_to_joint
                       if name in ("FR_calf_joint", "BR_calf_joint")
                       else calf_motor_to_joint)
            q, qd, ratio = convert(
                float(q_raw[name]),
                float(qd_raw[name]) if name in qd_raw else None,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            q_out.pop(name, None)
            qd_out.pop(name, None)
            warnings.append(f"{name}: {exc}")
            continue
        q_out[name] = q
        ratios[name] = ratio
        if qd is not None:
            qd_out[name] = qd
    return q_out, qd_out, ratios, warnings


def motor_to_sim_position(name: str, q_motor: float) -> float:
    """Hardware calibrated motor/crank A rad -> MuJoCo simulated joint rad."""
    if name in ("FR_calf_joint", "BR_calf_joint"):
        return mirrored_right_motor_to_joint(q_motor)[0]
    if name in ("FL_calf_joint", "BL_calf_joint"):
        return calf_motor_to_joint(q_motor)[0]
    return float(q_motor)


def sim_to_motor_position(name: str, q_sim: float) -> float:
    """Inverse four-bar IK by bisection on the specific hardware motor range.

    The Grallator four-bar is not a parallelogram (BC != AD). Do not merely
    send the XML rocker angle to the calf motor. The tested mechanical range
    for each calf maps monotonically to rocker angle on its chosen assembly
    branch. Return a bounded motor position in calibrated joint radians.
    """
    if not math.isfinite(q_sim):
        raise ValueError("nonfinite requested joint position")
    if name not in CALF_MOTOR_LIMITS_DEG:
        return float(q_sim)

    motor_min_deg, motor_max_deg = CALF_MOTOR_LIMITS_DEG[name]
    lo, hi = math.radians(motor_min_deg), math.radians(motor_max_deg)
    sim_lo = motor_to_sim_position(name, lo)
    sim_hi = motor_to_sim_position(name, hi)
    min_sim, max_sim = sorted((sim_lo, sim_hi))
    if not (min_sim - 1e-8 <= q_sim <= max_sim + 1e-8):
        raise ValueError(
            f"{name}: requested {math.degrees(q_sim):+.3f} deg outside "
            f"physical four-bar range "
            f"[{math.degrees(min_sim):+.3f}, "
            f"{math.degrees(max_sim):+.3f}] deg"
        )
    q_sim = max(min_sim, min(max_sim, q_sim))
    rising = sim_hi > sim_lo
    for _ in range(56):
        mid = (lo + hi) * 0.5
        fmid = motor_to_sim_position(name, mid)
        if (fmid < q_sim) == rising:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def hardware_and_xml_bounds(model, mapping, name):
    """Finite joint bounds; intersect with active XML limits."""
    m = mapping[name]
    jid = m["jid"]

    if name in CALF_MOTOR_LIMITS_DEG:
        motor_lo, motor_hi = CALF_MOTOR_LIMITS_DEG[name]
        lo, hi = sorted((
            motor_to_sim_position(name, math.radians(motor_lo)),
            motor_to_sim_position(name, math.radians(motor_hi)),
        ))
    elif name.endswith("_hip_joint"):
        lo, hi = -0.79, 0.79
    elif name.endswith("_thigh_joint"):
        lo, hi = -1.68, 1.68
    else:
        raise ValueError(f"{name}: unknown joint limits")

    if int(model.jnt_limited[jid]):
        xml_lo, xml_hi = float(m["lo"]), float(m["hi"])
        if not (math.isfinite(xml_lo) and math.isfinite(xml_hi)
                and xml_hi > xml_lo):
            raise ValueError(f"{name}: invalid XML limits")
        lo, hi = max(lo, xml_lo), min(hi, xml_hi)

    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"{name}: invalid bounded range")

    return lo, hi


def validate_fourbar_inverse():
    """Refuse control when specified motor windows cross a non-monotonic branch."""
    for name, (lower, upper) in CALF_MOTOR_LIMITS_DEG.items():
        last_value = None
        direction = None
        for motor_deg in np.linspace(lower, upper, 81):
            q = motor_to_sim_position(name, math.radians(float(motor_deg)))
            if last_value is not None:
                delta = q - last_value
                if abs(delta) < 1e-10:
                    raise ValueError(f"{name}: degenerate four-bar mapping")
                sign = 1 if delta > 0 else -1
                if direction is None:
                    direction = sign
                elif direction != sign:
                    raise ValueError(f"{name}: non-monotone four-bar branch")
            last_value = q




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
        self._meta = {}
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
                dict(self._meta),
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
                        if name in JOINTS and math.isfinite(float(value)):
                            q[name] = float(value)
                    if isinstance(packet.get("qd"), dict):
                        for name, value in packet["qd"].items():
                            if name in JOINTS and math.isfinite(float(value)):
                                qd[name] = float(value)

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
                self._meta = {
                    "selected_joint": packet.get("selected_joint", packet.get("controlled_joint")),
                    "bridge_state": packet.get("bridge_state", "UNKNOWN"),
                    "fault": packet.get("fault") if isinstance(packet.get("fault"), dict) else {},
                    "temperature": packet.get("temperature_c") if isinstance(packet.get("temperature_c"), dict) else {},
                }
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


def make_dashboard(data, mapping, state, age_ms, selected_joint):
    """Render MuJoCo qpos readback; not the original UDP motor encoder q."""
    lines = [
        "GRALLATOR | MUJOCO XML JOINT ANGLES (LIVE data.qpos)",
        f"Motor: {state}  |  selected: {selected_joint}  |  "
        f"feedback age: {age_ms:.0f}ms" if math.isfinite(age_ms)
        else f"Motor: {state}  |  selected: {selected_joint}  |  feedback: NO PACKETS",
        "=" * 50,
        f"{'JOINT':<18} {'DEG':>12} {'RAD':>12}",
        "-" * 50,
    ]
    for name in JOINTS:
        q = float(data.qpos[mapping[name]["qpos"]])
        lines.append(f"{name:<18} {math.degrees(q):+12.3f} {q:+12.5f}")
    lines.extend(["-" * 50, "Ctrl+C or close the viewer/panel to exit. MOTOR OFF by default."])
    return "\n".join(lines) + "\n"


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
        target = float(np.clip(q_des[name], *m["safe_bounds"]))
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


def end_to_end_measurements_mm(model, data, foot_map):
    """
    Return true 3D TIP-TO-TIP foot distances in centimeters.

    The tip definition is exactly the same one used by the green viewer points:
        tip = foot_center + radius * unit(foot_center - calf_center)

    Measurements:
        FRONT : FL tip <-> FR tip
        REAR  : BL tip <-> BR tip
        LEFT  : FL tip <-> BL tip
        RIGHT : FR tip <-> BR tip

    Full Euclidean distance is used:
        sqrt(dx^2 + dy^2 + dz^2)
    """

    points = get_foot_edge_points(
        model,
        data,
        foot_map,
    )

    def distance_3d_cm(a, b):
        delta = (
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )

        return float(
            np.linalg.norm(delta) * 100.0
        )

    FL = points["FL"]["front"]
    FR = points["FR"]["front"]
    BL = points["BL"]["front"]
    BR = points["BR"]["front"]

    front_width = distance_3d_cm(
        FL,
        FR,
    )

    rear_width = distance_3d_cm(
        BL,
        BR,
    )

    left_length = distance_3d_cm(
        FL,
        BL,
    )

    right_length = distance_3d_cm(
        FR,
        BR,
    )

    return {
        "front_width": front_width,
        "rear_width": rear_width,
        "left_length": left_length,
        "right_length": right_length,
    }

def print_end_to_end_measurements(model, data, foot_map):

    d = end_to_end_measurements_mm(
        model,
        data,
        foot_map,
    )

    print(
        f"FOOT TIP 3D [cm] | "
        f"FRONT={d['front_width']:7.2f} | "
        f"REAR={d['rear_width']:7.2f} | "
        f"LEFT={d['left_length']:7.2f} | "
        f"RIGHT={d['right_length']:7.2f}"
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

    Distance is full 3D XYZ Euclidean distance between GREEN foot-tip markers.
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
        True 3D Euclidean distance between GREEN foot-tip markers.
        """
        delta = (
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )

        return float(
            np.linalg.norm(delta) * 100.0
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



class JointSliderPanel:
    """Separate Tk window beside native MuJoCo viewer (same Python process)."""

    def __init__(self, joint_bounds, selected_joint, hardware_flag):
        self.root = tk.Tk()
        self.root.title("Grallator | MuJoCo motor controls")
        self.root.geometry("840x770")
        self.root.minsize(660, 520)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.closed = False
        self.selected = selected_joint
        self.hardware_flag = hardware_flag
        self.arm_request = None
        self.target_sim = None
        self._updating = False
        self.enable_var = tk.BooleanVar(self.root, value=False)
        self.status_var = tk.StringVar(self.root, value="MOTOR CONTROL OFF — readback only")

        bar = ttk.Frame(self.root, padding=12)
        bar.pack(fill="x")
        self.arm_checkbox = ttk.Checkbutton(
            bar, text="ENABLE REAL MOTOR CONTROL", variable=self.enable_var,
            command=self._arm_toggled,
        )
        self.arm_checkbox.pack(side="left")
        ttk.Button(bar, text="DISARM / STOP", command=self.force_off).pack(side="right")

        ttk.Label(
            self.root, textvariable=self.status_var, padding=(12, 2),
            foreground="#a64b00", wraplength=790,
        ).pack(fill="x")
        ttk.Label(
            self.root,
            text=f"Physical motor permitted: {selected_joint} only. "
                 "Other sliders track hardware feedback (read-only).",
            padding=(12, 0), wraplength=790,
        ).pack(fill="x")

        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, text="JOINT", width=21).pack(side="left")
        ttk.Label(header, text="XML JOINT SLIDER (DEGREES)").pack(side="left", expand=True)
        ttk.Label(header, text="TARGET", width=12).pack(side="right")
        ttk.Label(header, text="ACTUAL", width=12).pack(side="right")

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas, padding=(9, 3))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda ev: canvas.itemconfigure(window_id, width=ev.width))

        self.scales = {}
        self.actual_labels = {}
        self.target_labels = {}
        self.joint_bounds = joint_bounds
        for index, name in enumerate(JOINTS):
            if index % 3 == 0:
                ttk.Label(inner, text=f"{name[:2]} LEG", padding=(3, 10, 0, 3)).pack(fill="x")
            r = ttk.Frame(inner, padding=(3, 1))
            r.pack(fill="x")
            ttk.Label(r, text=name, width=21).pack(side="left")
            lo, hi = joint_bounds[name]
            act = tk.StringVar(self.root, value="n/a")
            tar = tk.StringVar(self.root, value="n/a")
            self.actual_labels[name] = act
            self.target_labels[name] = tar
            ttk.Label(r, textvariable=tar, width=12, anchor="e").pack(side="right")
            ttk.Label(r, textvariable=act, width=12, anchor="e").pack(side="right")
            slider = ttk.Scale(
                r, from_=math.degrees(lo), to=math.degrees(hi),
                orient="horizontal", length=280,
                command=lambda val, n=name: self._slider_changed(n, val),
            )
            slider.pack(fill="x", expand=True, padx=(4, 12), side="left")
            slider.state(["disabled"])
            slider.set(0.5 * math.degrees(lo + hi))
            self.scales[name] = slider

        ttk.Label(
            self.root, padding=(11, 6), wraplength=800,
            text="Calf sliders are virtual rocker angles (four-bar IK). "
                 "Real calf motor encoder limits and XML joint limits are both enforced. "
                 "Suspended-robot, one-joint test only; a screen toggle is not a physical E-stop.",
        ).pack(fill="x")
        self.refresh_control_states()

    def _close(self):
        self.force_off()
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _arm_toggled(self):
        self.arm_request = bool(self.enable_var.get())

    def _slider_changed(self, name, text_value):
        if self._updating or name != self.selected or not self.enable_var.get():
            return
        value_deg = float(text_value)
        self.target_sim = math.radians(value_deg)
        self.target_labels[name].set(f"{value_deg:+.2f} deg")

    def pop_arm_request(self):
        req, self.arm_request = self.arm_request, None
        return req

    def force_off(self, reason="MOTOR CONTROL OFF — readback only"):
        # A user-initiated stop queues one DISARM request. A disarm already
        # processed in the main loop must NOT continuously requeue itself.
        self.arm_request = False if self.enable_var.get() else None
        self.enable_var.set(False)
        self.target_sim = None
        self.status_var.set(reason)
        self.refresh_control_states()

    def set_status(self, text):
        self.status_var.set(text)

    def refresh_control_states(self):
        for name, slider in self.scales.items():
            if self.enable_var.get() and self.hardware_flag and name == self.selected:
                slider.state(["!disabled"])
            else:
                slider.state(["disabled"])

    def set_target(self, name, q_sim):
        self._updating = True
        try:
            self.scales[name].set(math.degrees(q_sim))
            self.target_labels[name].set(f"{math.degrees(q_sim):+.2f} deg")
        finally:
            self._updating = False
        if name == self.selected:
            self.target_sim = float(q_sim)

    def refresh_feedback(self, sim_q):
        self._updating = True
        try:
            for name, q in sim_q.items():
                if name not in self.scales:
                    continue
                self.actual_labels[name].set(f"{math.degrees(q):+.2f} deg")
                if name != self.selected or not self.enable_var.get():
                    self.scales[name].set(math.degrees(q))
                    self.target_labels[name].set(f"{math.degrees(q):+.2f} deg")
        finally:
            self._updating = False

    def update(self):
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True


def make_motor_command(session, seq, armed, selected_joint, q_des_motor):
    """Exactly the packet accepted by existing Jetson one-joint UDP receiver."""
    if not math.isfinite(q_des_motor):
        raise ValueError("nonfinite motor command")
    return {
        "type": "grallator_mujoco_command_v1",
        "session": session,
        "seq": int(seq),
        "armed": bool(armed),
        "joint": selected_joint,
        "q_des": {selected_joint: float(q_des_motor)},
        "timestamp_pc": time.time(),
    }


def arm_inhibit_reason(args, fresh, source, metadata, q_real, q_joint, qd_real, mapping):
    """PC-side safety preconditions; Jetson still owns final motor authority."""
    name = args.control_joint
    if not args.enable_hardware_command:
        return "Run with --enable-hardware-command for physical motor commands"
    if not fresh or name not in q_real or name not in q_joint:
        return "Missing or stale selected-joint encoder feedback"
    if source is None or source[0] != args.jetson_resolved_ip:
        return "Telemetry source does not match the Jetson host"
    if metadata.get("selected_joint") != name:
        return "Jetson selected_joint mismatch or not present; v1 receiver required"
    if metadata.get("fault", {}).get(name, 0):
        return f"Jetson reports motor fault on {name}"
    state = str(metadata.get("bridge_state", "UNKNOWN"))
    if any(k in state for k in ("ESTOP", "FAULT", "TORQUE_LIMIT", "OVER_TEMPERATURE")):
        return f"Unsafe Jetson state: {state}"
    q_motor = float(q_real[name])
    if not math.isfinite(q_motor):
        return "Nonfinite selected-joint encoder value"
    if name in CALF_MOTOR_LIMITS_DEG:
        lo, hi = map(math.radians, CALF_MOTOR_LIMITS_DEG[name])
        if not (lo <= q_motor <= hi):
            return f"{name} encoder outside supplied physical motor limits"
    lo, hi = mapping[name]["safe_bounds"]
    if not (lo - 1e-4 <= q_joint[name] <= hi + 1e-4):
        return "Measured angle outside the XML/physical joint-safe bounds"
    if name not in qd_real:
        return "Missing selected-joint velocity feedback from Jetson"
    velocity = float(qd_real[name])
    if not math.isfinite(velocity) or abs(velocity) > 0.20:
        return "Joint must be nearly stationary (|qd| <= 0.20 rad/s)"
    return None


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=here / "model" / "Grallator_Final.xml")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=57543,
                        help="Incoming Jetson joint feedback UDP port")
    parser.add_argument("--mode", choices=("direct", "pd"), default="direct")
    parser.add_argument("--kp", type=float, default=250.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--torque-limit", type=float, default=100.0)
    parser.add_argument("--stale-ms", type=float, default=250.0)
    parser.add_argument(
        "--no-fourbar", action="store_true",
        help="Disable four-bar calf conversion for receive-only diagnostics; hardware command blocked.",
    )
    parser.add_argument("--foot-measure-hz", type=float, default=0.0,
                        help="Deprecated; foot distances displayed in MuJoCo viewer only.")
    parser.add_argument("--show-foot-points", action="store_true")
    parser.add_argument("--foot-point-radius", type=float, default=0.010)

    parser.add_argument("--enable-hardware-command", action="store_true",
                        help="Allow GUI master toggle to send real hardware motor commands. OFF by default.")
    parser.add_argument("--control-joint", choices=JOINTS, default="FR_thigh_joint",
                        help="Must match the ONE selected joint on the Jetson v1 receiver")
    parser.add_argument("--jetson-host", default="192.168.1.62")
    parser.add_argument("--command-port", type=int, default=57544)
    parser.add_argument("--command-hz", type=float, default=50.0)
    parser.add_argument("--pc-max-offset-rad", type=float, default=0.05,
                        help="Max selected motor displacement about position captured at ARM")
    args = parser.parse_args()

    if args.enable_hardware_command and args.no_fourbar:
        parser.error("--no-fourbar is diagnostic-only and cannot enable hardware command")
    if args.enable_hardware_command and args.mode != "direct":
        parser.error("For physical single-joint tests use --mode direct")
    if args.command_hz <= 0 or args.command_hz > 100:
        parser.error("--command-hz must be > 0 and <= 100")
    if not (0 < args.pc_max_offset_rad <= 0.05):
        parser.error("Stage-1 --pc-max-offset-rad must be within (0, 0.05]")
    if args.stale_ms <= 0 or args.stale_ms > 250:
        parser.error("--stale-ms must be > 0 and <= 250 when using this bridge")
    if not (0.0 < args.foot_point_radius < 0.1):
        parser.error("Invalid foot point marker radius")

    args.jetson_resolved_ip = socket.gethostbyname(args.jetson_host)
    validate_fourbar_inverse()

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data)
    mapping = build_map(model)
    for name in JOINTS:
        # Hard limiter used both by UI and physical commands.
        mapping[name]["safe_bounds"] = hardware_and_xml_bounds(model, mapping, name)
    foot_map = build_foot_geom_map(model)

    panel = JointSliderPanel(
        {name: mapping[name]["safe_bounds"] for name in JOINTS},
        args.control_joint, args.enable_hardware_command,
    )
    receiver = LatestJointPacket(args.bind, args.port)
    receiver.start()

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.enable_hardware_command else None
    destination = (args.jetson_resolved_ip, args.command_port)
    session = uuid.uuid4().hex
    seq = 0
    arm_reference_motor = None
    armed = False
    last_motor_cmd = None
    next_command = time.monotonic()
    next_gui_update = 0.0
    next_terminal_update = 0.0
    sim_dt = float(model.opt.timestep)
    is_tty = sys.stdout.isatty()

    def send_command(enabled, position):
        nonlocal seq
        if cmd_sock is None:
            return
        packet = make_motor_command(session, seq, enabled, args.control_joint, position)
        cmd_sock.sendto(json.dumps(packet, separators=(",", ":")).encode(), destination)
        seq += 1

    def disarm(reason):
        nonlocal armed, arm_reference_motor, last_motor_cmd
        armed = False
        arm_reference_motor = None
        last_motor_cmd = None
        panel.force_off(reason)

    print(f"XML: {args.xml} | Feedback :{args.port} | Jetson: {args.jetson_host}:{args.command_port}")
    print("Master enable OFF. Other motors never commanded by the stage-1 receiver.")

    try:
        if is_tty:
            # Alternate terminal buffer: normal shell history is untouched.
            sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")
            sys.stdout.flush()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not panel.closed:
                loop_start = time.monotonic()
                q_real, qd_real, last_rx, source, packet_count, meta = receiver.snapshot()
                now = time.monotonic()
                age = float("inf") if last_rx is None else 1000.0 * (now - last_rx)
                fresh = last_rx is not None and age <= args.stale_ms

                if args.no_fourbar:
                    q_joint, qd_joint = q_real, qd_real
                else:
                    q_joint, qd_joint, _, _ = convert_calf_packet(q_real, qd_real)

                if now >= next_gui_update:
                    panel.refresh_feedback(q_joint)
                    panel.refresh_control_states()
                    panel.update()
                    next_gui_update = now + 0.05

                enable_req = panel.pop_arm_request()
                if enable_req is False:
                    disarm("MOTOR OFF — GUI toggle/stop pressed")
                elif enable_req is True:
                    reason = arm_inhibit_reason(
                        args, fresh, source, meta, q_real, q_joint, qd_real, mapping
                    )
                    if reason:
                        disarm("ARM REFUSED: " + reason)
                    else:
                        # No-step initial target, captured from REAL encoder.
                        arm_reference_motor = float(q_real[args.control_joint])
                        last_motor_cmd = arm_reference_motor
                        panel.set_target(args.control_joint, q_joint[args.control_joint])
                        armed = True
                        panel.set_status(
                            f"ARMED (request) {args.control_joint} | "
                            f"limit +/-{math.degrees(args.pc_max_offset_rad):.2f} deg from ARM motor pose"
                        )

                if armed:
                    reason = arm_inhibit_reason(
                        args, fresh, source, meta, q_real, q_joint, qd_real, mapping
                    )
                    if reason:
                        disarm("AUTO-DISARM: " + reason)

                if armed:
                    name = args.control_joint
                    q_des_sim = panel.target_sim
                    if q_des_sim is None:
                        disarm("AUTO-DISARM: lost active slider target")
                    else:
                        # Clamp rocker/virtual angle to intersection of XML and
                        # physical four-bar limits, then invert it to motor angle.
                        safe_lo, safe_hi = mapping[name]["safe_bounds"]
                        q_des_sim = max(safe_lo, min(safe_hi, float(q_des_sim)))
                        try:
                            motor_des = sim_to_motor_position(name, q_des_sim)
                            motor_des = float(np.clip(
                                motor_des,
                                arm_reference_motor - args.pc_max_offset_rad,
                                arm_reference_motor + args.pc_max_offset_rad,
                            ))
                            if name in CALF_MOTOR_LIMITS_DEG:
                                motor_min, motor_max = CALF_MOTOR_LIMITS_DEG[name]
                                motor_des = float(np.clip(
                                    motor_des, math.radians(motor_min), math.radians(motor_max),
                                ))
                            if not math.isfinite(motor_des):
                                raise ValueError("nonfinite motor target")
                            last_motor_cmd = motor_des
                            # Reflect the safety-clamped target back onto slider.
                            clamped_sim = motor_to_sim_position(name, motor_des)
                            if abs(clamped_sim - q_des_sim) > 1e-7:
                                panel.set_target(name, clamped_sim)
                        except (ValueError, OverflowError) as exc:
                            disarm("AUTO-DISARM: invalid target: " + str(exc))

                # Visualization always reflects measured robot when default
                # --mode direct. In PD mode, sim integrates dynamics as before.
                with viewer.lock():
                    if args.mode == "direct":
                        if q_joint:
                            apply_direct(model, data, mapping, q_joint, qd_joint)
                    else:
                        if fresh:
                            apply_pd(model, data, mapping, q_joint, qd_joint,
                                     args.kp, args.kd, args.torque_limit)
                        else:
                            data.ctrl[:] = 0.0
                        mujoco.mj_step(model, data)
                    if args.show_foot_points:
                        draw_foot_edge_markers(
                            viewer, model, data, foot_map, args.foot_point_radius
                        )
                viewer.sync()

                if cmd_sock is not None and now >= next_command:
                    # Disarmed packets are safe heartbeats. Receiver's own
                    # watchdog STOPs the motor on command/feedback loss.
                    fallback = float(q_real.get(args.control_joint, 0.0))
                    safe_position = last_motor_cmd if armed and last_motor_cmd is not None else fallback
                    try:
                        send_command(armed, safe_position)
                    except OSError as exc:
                        disarm("AUTO-DISARM: UDP command send failed: " + str(exc))
                    next_command += 1.0 / args.command_hz
                    if next_command < now - (1.0 / args.command_hz):
                        next_command = now + (1.0 / args.command_hz)

                if now >= next_terminal_update:
                    state = "ARMED REQUEST" if armed else "OFF"
                    if not fresh:
                        state += " / STALE"
                    text = make_dashboard(data, mapping, state, age, args.control_joint)
                    if is_tty:
                        sys.stdout.write("\x1b[H\x1b[2J" + text)
                    else:
                        # Don't spam logs when run without an interactive TTY.
                        pass
                    sys.stdout.flush()
                    next_terminal_update = now + 0.10

                elapsed = time.monotonic() - loop_start
                if elapsed < (sim_dt if args.mode == "pd" else 0.005):
                    time.sleep((sim_dt if args.mode == "pd" else 0.005) - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        # Send multiple disarmed messages with increasing seq to survive UDP
        # packet loss. The Jetson MUST also enforce its own watchdog stop.
        if cmd_sock is not None:
            for _ in range(5):
                try:
                    send_command(False, float(last_motor_cmd or 0.0))
                except OSError:
                    pass
                time.sleep(0.005)
            cmd_sock.close()
        receiver.close()
        if not panel.closed:
            panel._close()
        if is_tty:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
