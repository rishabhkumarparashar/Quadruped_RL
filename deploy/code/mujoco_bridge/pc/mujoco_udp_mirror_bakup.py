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
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data)
    mapping = build_map(model)

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

                viewer.sync()

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
