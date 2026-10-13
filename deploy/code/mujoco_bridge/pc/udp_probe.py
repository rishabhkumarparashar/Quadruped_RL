#!/usr/bin/env python3
"""Receive Grallator joint telemetry from the Jetson and print packet rate."""

import argparse
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=57543)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(1.0)

    print(f"Listening for Jetson UDP on {args.bind}:{args.port}")
    print("Move a real joint by hand. Ctrl+C to stop.\n")

    count = 0
    last_report = time.monotonic()

    while True:
        try:
            raw, source = sock.recvfrom(65535)
        except socket.timeout:
            print("No packet received in the last 1 s")
            continue

        packet = json.loads(raw.decode("utf-8"))
        count += 1

        now = time.monotonic()
        if now - last_report >= 1.0:
            joints = packet.get("joints", [])
            q = {
                j.get("n"): j.get("qf")
                for j in joints
                if j.get("n") and j.get("qf") is not None
            }
            compact = "  ".join(
                f"{name}={value:+.3f}"
                for name, value in list(q.items())[:12]
            )
            print(f"from={source[0]}  rate≈{count/(now-last_report):.1f} Hz  {compact}")
            count = 0
            last_report = now


if __name__ == "__main__":
    main()
