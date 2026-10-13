#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$HERE/mujoco_hw_controller.py" \
  --xml "$HERE/model/Grallator_Final.xml" \
  --jetson-host 192.168.1.62 \
  --feedback-port 57543 \
  --command-port 57544 \
  --joint FR_thigh_joint \
  --command-hz 50 \
  --step-rad 0.01 \
  --pc-max-offset-rad 0.05 \
  "$@"
