#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/mujoco_udp_mirror_backup_2.py" --mode direct --port 57543
