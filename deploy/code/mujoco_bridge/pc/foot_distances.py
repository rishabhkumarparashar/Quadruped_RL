#!/usr/bin/env python3
"""Calculate Grallator foot-centre distances from 12 encoder readings.

Run: python3 foot_distances.py
Enter three angles per leg: hip, thigh, calf MOTOR encoder, in DEGREES.
Use the same signed readings sent to the existing MuJoCo bridge.

If entering the MuJoCo joint dashboard values instead, run:
    python3 foot_distances.py --joint-angles

Only Python's standard library is needed. No robot connection is made.
Dimensions below are copied from model/Grallator_Final.xml on 2026-10-11.
This predicts the model's distances using the bridge's existing zero/sign
assumptions; it does not correct the unresolved hardware calibration error.
"""

import argparse
import math


# All positions are in MILLIMETRES: (forward X, left Y, upward Z).
# Each vector is measured in its parent link's frame at joint angle zero.
GEOMETRY = {
    "FL": {
        "body_to_hip": (392.5, 101.89, 79.667),
        "hip_to_thigh": (105.613884755068, 1.11315821878633, -5.98992399467524),
        "thigh_to_knee": (-172.83, 131.0, -291.6),
        "knee_to_foot": (174.71, -0.1, -310.57),
    },
    "FR": {
        "body_to_hip": (392.5, -103.3, 78.238),
        "hip_to_thigh": (104.55, 0.024752, -5.2806),
        "thigh_to_knee": (-171.77, -131.0, -290.88),
        "knee_to_foot": (174.71, 0.1, -310.57),
    },
    "BL": {
        "body_to_hip": (-392.5, 101.89, 73.13),
        "hip_to_thigh": (-100.38, 1.1132, 0.55515),
        "thigh_to_knee": (-172.84, 131.0, -291.61),
        "knee_to_foot": (174.71, -0.1, -310.57),
    },
    "BR": {
        "body_to_hip": (-392.5, -103.27, 74.508),
        "hip_to_thigh": (-100.38, -1.113, 0.555),
        "thigh_to_knee": (-172.84, -131.0, -291.61),
        "knee_to_foot": (174.71, -0.1, -310.57),
    },
}

# Joint axes from the XML: signs around X (hip) and Y (thigh/calf).
# These are applied AFTER reading the signed encoder values.
AXIS_SIGNS = {
    "FL": (1, 1, -1),
    "FR": (1, -1, 1),
    "BL": (-1, 1, -1),
    "BR": (-1, -1, 1),
}

# Calf four-bar link lengths in mm, matching the existing bridge.
AB = 55.30                  # Motor crank
BC = 343.67                 # Connecting rod
CD = 55.30                  # Knee rocker
AD = 340.00                 # Fixed base
CRANK_ZERO = math.radians(134.29)


def rocker_angle(crank_angle):
    """Solve the four-bar on the same assembly branch as the bridge."""
    bx = AB * math.cos(crank_angle)
    by = AB * math.sin(crank_angle)
    bd = math.hypot(bx - AD, by)
    if bd <= 1e-10 or bd < abs(BC - CD) - 1e-8 or bd > BC + CD + 1e-8:
        raise ValueError("The four-bar cannot close at this motor angle.")

    cosine = (bd**2 + CD**2 - BC**2) / (2 * bd * CD)
    alpha = math.acos(max(-1.0, min(1.0, cosine)))
    beta = math.atan2(by, bx - AD)
    return beta - alpha


ROCKER_ZERO = rocker_angle(CRANK_ZERO)


def calf_motor_to_joint(leg, motor_angle):
    """Convert calf motor angle to calf joint angle; both are radians."""
    if leg in ("FL", "BL"):
        joint_angle = ROCKER_ZERO - rocker_angle(CRANK_ZERO - motor_angle)
    else:
        # Right-side reflected geometry, simplified from the bridge.
        joint_angle = rocker_angle(CRANK_ZERO + motor_angle) - ROCKER_ZERO

    return math.atan2(math.sin(joint_angle), math.cos(joint_angle))


def add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def rotate_x(point, angle):
    x, y, z = point
    c, s = math.cos(angle), math.sin(angle)
    return (x, c * y - s * z, s * y + c * z)


def rotate_y(point, angle):
    x, y, z = point
    c, s = math.cos(angle), math.sin(angle)
    return (c * x + s * z, y, -s * x + c * z)


def foot_position(leg, angles_deg, motor_encoders=True):
    """Return this foot centre's (X, Y, Z) in mm relative to the body.

    angles_deg contains (hip, thigh, calf) in degrees.
    """
    if len(angles_deg) != 3 or not all(math.isfinite(a) for a in angles_deg):
        raise ValueError(f"{leg}: enter three finite angles.")
    hip, thigh, calf = [math.radians(a) for a in angles_deg]
    if motor_encoders:
        calf = calf_motor_to_joint(leg, calf)

    hip_sign, thigh_sign, calf_sign = AXIS_SIGNS[leg]
    g = GEOMETRY[leg]

    # Start at the foot and work back up the leg toward the body.
    point = rotate_y(g["knee_to_foot"], calf_sign * calf)
    point = add(g["thigh_to_knee"], point)
    point = rotate_y(point, thigh_sign * thigh)
    point = add(g["hip_to_thigh"], point)
    point = rotate_x(point, hip_sign * hip)
    return add(g["body_to_hip"], point)


def foot_distances_cm(angles_deg, motor_encoders=True):
    """Calculate the four full 3D centre-to-centre distances in cm."""
    feet = {
        leg: foot_position(leg, angles_deg[leg], motor_encoders)
        for leg in GEOMETRY
    }
    pairs = {
        "FRONT (FL-FR)": ("FL", "FR"),
        "REAR  (BL-BR)": ("BL", "BR"),
        "LEFT  (FL-BL)": ("FL", "BL"),
        "RIGHT (FR-BR)": ("FR", "BR"),
    }
    # Positions are mm, so divide their Euclidean distance by 10 for cm.
    return {name: math.dist(feet[a], feet[b]) / 10 for name, (a, b) in pairs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-angles", action="store_true",
        help="Input MuJoCo joint angles; skip the calf motor conversion.",
    )
    args = parser.parse_args()
    source = "MuJoCo JOINT angles" if args.joint_angles else "MOTOR ENCODER angles"
    print(f"Enter {source} in DEGREES: hip thigh calf")
    print("Keep the original signs. Example: -47.18 -2.79 -20.72\n")

    angles = {}
    for leg in GEOMETRY:
        while True:
            try:
                values = [float(v) for v in input(f"{leg}: ").replace(",", " ").split()]
                # Validate this leg before accepting its inputs.
                foot_position(leg, values, motor_encoders=not args.joint_angles)
                angles[leg] = values
                break
            except ValueError as error:
                print(f"Invalid {leg} input: {error} Please try again.")

    distances = foot_distances_cm(angles, motor_encoders=not args.joint_angles)
    print("\nFOOT CENTRE-TO-CENTRE DISTANCES (3D)")
    for name, distance in distances.items():
        print(f"{name}: {distance:8.2f} cm")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nInput cancelled.")
