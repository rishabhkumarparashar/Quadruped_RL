#!/usr/bin/env python3
"""
GRALLATOR MuJoCo -> real-hardware command console (PC side)

FIRST-STAGE PURPOSE
-------------------
This is intentionally a ONE-JOINT test controller.

It:
  1. Receives calibrated real joint feedback from the Jetson on UDP 57543.
  2. Loads Grallator_Final.xml and uses the MuJoCo joint-name -> qpos mapping.
  3. Mirrors q_real into MuJoCo for visualization.
  4. Lets you ARM/DISARM from the PC terminal.
  5. Lets you nudge ONE selected joint target by a very small amount.
  6. Sends q_des to the Jetson on UDP 57544 at 50 Hz.

The Jetson remains responsible for:
  - CAN
  - MIT commands
  - Kp/Kd
  - command-rate limiting
  - torque limiting
  - encoder freshness
  - motor fault/temperature checks
  - watchdog stop

Keys
----
  a        arm (target is FIRST reset to the live measured joint angle)
  d        disarm
  ] or +   increase selected joint target by --step-rad
  [ or -   decrease selected joint target by --step-rad
  r        reset target to current measured joint angle
  v        toggle viewer between REAL and TARGET preview
  q / x    disarm and quit

IMPORTANT
---------
Run the matching Jetson mujoco_command_receiver.py instead of a second
CAN polling/control program. Do not run another program that can send
stop/enable/MIT frames to the same motor during this test.
"""

import argparse
import json
import select
import socket
import sys
import termios
import threading
import time
import tty
import uuid
from pathlib import Path

import mujoco
import mujoco.viewer


JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "BR_hip_joint", "BR_thigh_joint", "BR_calf_joint",
    "BL_hip_joint", "BL_thigh_joint", "BL_calf_joint",
]


class LatestTelemetry:
    def __init__(self, bind_host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_host, port))
        self.sock.settimeout(0.20)
        self.lock = threading.Lock()
        self.q = {}
        self.qd = {}
        self.last_rx = None
        self.source = None
        self.count = 0
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_flag = True
        try:
            self.sock.close()
        except OSError:
            pass

    def snapshot(self):
        with self.lock:
            return (
                dict(self.q),
                dict(self.qd),
                self.last_rx,
                self.source,
                self.count,
            )

    def _run(self):
        while not self.stop_flag:
            try:
                raw, source = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                packet = json.loads(raw.decode("utf-8"))
                q = {}
                qd = {}

                # New bridge packet:
                # {"q": {...}, "qd": {...}}
                if isinstance(packet.get("q"), dict):
                    for name, value in packet["q"].items():
                        if name in JOINTS:
                            q[name] = float(value)
                    if isinstance(packet.get("qd"), dict):
                        for name, value in packet["qd"].items():
                            if name in JOINTS:
                                qd[name] = float(value)

                # Compatibility with the older "joints" telemetry format.
                elif isinstance(packet.get("joints"), list):
                    for item in packet["joints"]:
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

            with self.lock:
                self.q = q
                self.qd = qd
                self.last_rx = time.monotonic()
                self.source = source
                self.count += 1


class TerminalKeys:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.old = None

    def __enter__(self):
        if self.enabled:
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)

    def read(self):
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)


def name_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return idx


def build_joint_map(model):
    mapping = {}
    for name in JOINTS:
        jid = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mapping[name] = {
            "jid": jid,
            "qpos": int(model.jnt_qposadr[jid]),
            "dof": int(model.jnt_dofadr[jid]),
        }
    return mapping


def apply_view_state(model, data, mapping, q_real, qd_real, q_target, view_target, base_z):
    for name, m in mapping.items():
        if name not in q_real:
            continue

        if view_target and name in q_target:
            data.qpos[m["qpos"]] = float(q_target[name])
            data.qvel[m["dof"]] = 0.0
        else:
            data.qpos[m["qpos"]] = float(q_real[name])
            data.qvel[m["dof"]] = float(qd_real.get(name, 0.0))

    # Visualization-only floating-base height.
    # The 12 joint angles still come from real hardware.
    # This does NOT send any base command to the robot.
    data.qpos[2] = float(base_z)

    mujoco.mj_forward(model, data)


def make_packet(session, seq, armed, joint, q_des):
    return {
        "type": "grallator_mujoco_command_v1",
        "session": session,
        "seq": int(seq),
        "armed": bool(armed),
        "joint": joint,
        "q_des": {joint: float(q_des)},
        "timestamp_pc": time.time(),
    }


def send_packet(sock, dest, packet):
    raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
    sock.sendto(raw, dest)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=Path,
        default=here / "model" / "Grallator_Final.xml",
    )
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--feedback-port", type=int, default=57543)

    parser.add_argument("--jetson-host", default="192.168.1.62")
    parser.add_argument("--command-port", type=int, default=57544)
    parser.add_argument("--command-hz", type=float, default=50.0)

    parser.add_argument("--joint", choices=JOINTS, default="FR_thigh_joint")
    parser.add_argument("--step-rad", type=float, default=0.01)
    parser.add_argument(
        "--base-z",
        type=float,
        default=0.60,
        help="Visualization-only MuJoCo floating-base height in meters.",
    )
    parser.add_argument(
        "--pc-max-offset-rad",
        type=float,
        default=0.05,
        help="PC-side target clamp around the joint angle captured at ARM time.",
    )
    parser.add_argument(
        "--feedback-stale-ms",
        type=float,
        default=250.0,
        help="Auto-disarm if Jetson telemetry is older than this.",
    )
    args = parser.parse_args()

    if args.command_hz <= 0:
        parser.error("--command-hz must be > 0")
    if args.step_rad <= 0:
        parser.error("--step-rad must be > 0")
    if args.pc_max_offset_rad <= 0:
        parser.error("--pc-max-offset-rad must be > 0")

    xml_path = args.xml.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mapping = build_joint_map(model)

    # Put the model above the floor immediately when the viewer opens.
    data.qpos[2] = float(args.base_z)

    mujoco.mj_forward(model, data)

    telemetry = LatestTelemetry(args.bind, args.feedback_port)
    telemetry.start()

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.jetson_host, args.command_port)

    session = uuid.uuid4().hex
    seq = 0
    armed = False
    arm_reference = None
    q_target = {}
    view_target = False
    last_print = 0.0

    period = 1.0 / args.command_hz
    next_send = time.monotonic()

    print("=" * 78)
    print("GRALLATOR MUJOCO -> REAL HARDWARE : ONE-JOINT PC CONSOLE")
    print("=" * 78)
    print("XML             :", xml_path)
    print("Selected joint  :", args.joint)
    print("Feedback UDP    :", f"{args.bind}:{args.feedback_port}")
    print("Jetson command  :", f"{args.jetson_host}:{args.command_port}")
    print("Command rate    :", f"{args.command_hz:.1f} Hz")
    print("Nudge step      :", f"{args.step_rad:.4f} rad")
    print("PC max offset   :", f"+/-{args.pc_max_offset_rad:.4f} rad from ARM pose")
    print("Viewer base Z   :", f"{args.base_z:.3f} m")
    print()
    print("Keys: a=ARM  d=DISARM  ]/+ = +step  [/- = -step")
    print("      r=reset target to measured  v=real/target view  q/x=quit")
    print()
    print("Waiting for Jetson feedback...")

    try:
        with TerminalKeys() as keys:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                running = True
                while running and viewer.is_running():
                    loop_start = time.monotonic()
                    q_real, qd_real, last_rx, source, packet_count = telemetry.snapshot()

                    now = time.monotonic()
                    feedback_age = None if last_rx is None else now - last_rx
                    fresh = (
                        feedback_age is not None
                        and feedback_age <= args.feedback_stale_ms / 1000.0
                        and args.joint in q_real
                    )

                    if args.joint in q_real and args.joint not in q_target:
                        q_target[args.joint] = float(q_real[args.joint])

                    key = keys.read()
                    if key is not None:
                        if key in ("q", "Q", "x", "X"):
                            armed = False
                            running = False

                        elif key in ("d", "D"):
                            armed = False
                            print("\n[PC] DISARM")

                        elif key in ("a", "A"):
                            if not fresh:
                                print("\n[PC] ARM refused: selected joint feedback is missing/stale.")
                            else:
                                # Critical no-step arm behavior:
                                # q_des(arm) = q_real(arm)
                                arm_reference = float(q_real[args.joint])
                                q_target[args.joint] = arm_reference
                                armed = True
                                print(
                                    f"\n[PC] ARM requested. "
                                    f"{args.joint} reference={arm_reference:+.5f} rad"
                                )

                        elif key in ("]", "+", "="):
                            if args.joint in q_target:
                                proposed = q_target[args.joint] + args.step_rad
                                if arm_reference is not None:
                                    proposed = max(
                                        arm_reference - args.pc_max_offset_rad,
                                        min(
                                            arm_reference + args.pc_max_offset_rad,
                                            proposed,
                                        ),
                                    )
                                q_target[args.joint] = proposed
                                print(
                                    f"\n[PC] target {args.joint}="
                                    f"{q_target[args.joint]:+.5f} rad"
                                )

                        elif key in ("[", "-", "_"):
                            if args.joint in q_target:
                                proposed = q_target[args.joint] - args.step_rad
                                if arm_reference is not None:
                                    proposed = max(
                                        arm_reference - args.pc_max_offset_rad,
                                        min(
                                            arm_reference + args.pc_max_offset_rad,
                                            proposed,
                                        ),
                                    )
                                q_target[args.joint] = proposed
                                print(
                                    f"\n[PC] target {args.joint}="
                                    f"{q_target[args.joint]:+.5f} rad"
                                )

                        elif key in ("r", "R"):
                            if args.joint in q_real:
                                q_target[args.joint] = float(q_real[args.joint])
                                if armed:
                                    arm_reference = float(q_real[args.joint])
                                print(
                                    f"\n[PC] target reset to measured: "
                                    f"{q_target[args.joint]:+.5f} rad"
                                )

                        elif key in ("v", "V"):
                            view_target = not view_target
                            print(
                                "\n[PC] viewer = "
                                + ("TARGET preview" if view_target else "REAL feedback")
                            )

                    # Network/telemetry loss -> PC stops asking for arm.
                    if armed and not fresh:
                        armed = False
                        print("\n[PC] AUTO-DISARM: feedback became stale.")

                    if q_real:
                        apply_view_state(
                            model=model,
                            data=data,
                            mapping=mapping,
                            q_real=q_real,
                            qd_real=qd_real,
                            q_target=q_target,
                            view_target=view_target,
                            base_z=args.base_z,
                        )
                    viewer.sync()

                    if now >= next_send:
                        # If target is not initialized yet, send disarmed heartbeat.
                        q_des = float(
                            q_target.get(
                                args.joint,
                                q_real.get(args.joint, 0.0),
                            )
                        )
                        packet = make_packet(
                            session=session,
                            seq=seq,
                            armed=armed,
                            joint=args.joint,
                            q_des=q_des,
                        )
                        send_packet(cmd_sock, destination, packet)
                        seq += 1
                        next_send += period
                        if next_send < now - period:
                            next_send = now + period

                    if now - last_print >= 0.5:
                        last_print = now
                        q_txt = (
                            "n/a"
                            if args.joint not in q_real
                            else f"{q_real[args.joint]:+.5f}"
                        )
                        qd_txt = (
                            "n/a"
                            if args.joint not in qd_real
                            else f"{qd_real[args.joint]:+.5f}"
                        )
                        target_txt = (
                            "n/a"
                            if args.joint not in q_target
                            else f"{q_target[args.joint]:+.5f}"
                        )
                        age_txt = (
                            "n/a"
                            if feedback_age is None
                            else f"{1000.0 * feedback_age:.1f}ms"
                        )
                        print(
                            f"\rstate={'ARMED' if armed else 'DISARMED':8s} "
                            f"q={q_txt:>10s} qd={qd_txt:>10s} "
                            f"target={target_txt:>10s} age={age_txt:>9s} "
                            f"rx={packet_count:7d}",
                            end="",
                            flush=True,
                        )

                    elapsed = time.monotonic() - loop_start
                    if elapsed < 0.005:
                        time.sleep(0.005 - elapsed)

    finally:
        # Send several explicit DISARM packets so one lost UDP packet is not
        # the only thing separating the receiver from its watchdog.
        try:
            q_real, _, _, _, _ = telemetry.snapshot()
            q_des = float(
                q_target.get(args.joint, q_real.get(args.joint, 0.0))
            )
            for _ in range(5):
                send_packet(
                    cmd_sock,
                    destination,
                    make_packet(
                        session=session,
                        seq=seq,
                        armed=False,
                        joint=args.joint,
                        q_des=q_des,
                    ),
                )
                seq += 1
                time.sleep(0.02)
        except Exception:
            pass

        telemetry.close()
        cmd_sock.close()
        print("\n[PC] Closed. DISARM sent.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
