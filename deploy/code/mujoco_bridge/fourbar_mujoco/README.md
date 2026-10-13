# Grallator Four-Bar MuJoCo Hardware Mirror

Exact link dimensions used:

- AB crank = 55.30 mm
- BC coupler = 343.67 mm
- CD rocker = 55.30 mm
- AD ground = 340.00 mm

## Files

- `grallator_fourbar.xml` — closed-loop MuJoCo four-bar
- `grallator_fourbar_udp_mirror.py` — read-only UDP hardware mirror

## Safety

This script is intentionally one-way:

REAL ENCODER -> UDP -> MUJOCO

It does not send motor commands.

## Install

```bash
pip install mujoco
```

## Run

If your encoder broadcaster sends to UDP port 5005:

```bash
python3 grallator_fourbar_udp_mirror.py \
  --port 5005 \
  --joint FR_calf_joint \
  --geom-angle-at-start 60
```

If a different joint is being tested:

```bash
python3 grallator_fourbar_udp_mirror.py \
  --port 5005 \
  --joint FL_calf_joint \
  --geom-angle-at-start 60
```

## Critical calibration

`--geom-angle-at-start` must be the REAL physical angle between:

- ground link AD
- crank link AB

at the instant the first valid UDP packet arrives.

The script intentionally captures the first encoder value as the motor reference.
It then applies only encoder DELTA to the physical geometric reference angle.

Do not assume encoder zero is geometric zero.

With these exact dimensions, geometric crank angle 0 degrees cannot close.
The linkage becomes geometrically valid at approximately +/-19.253 degrees
from the AD axis, depending on assembly branch.

## Clockwise / anticlockwise test

Start at a known physical pose and note terminal output.

Move the motor:

- +10 deg
- +20 deg
- +30 deg
- back to reference
- -10 deg
- -20 deg
- -30 deg

The terminal shows:

- real encoder angle
- crank movement
- rocker absolute angle
- rocker movement
- average rocker/crank ratio

If the simulation moves opposite to the real crank convention, run with:

```bash
--encoder-sign -1
```

Use that only to match the chosen physical/geometric sign convention; it does
not alter the mechanism dimensions.
