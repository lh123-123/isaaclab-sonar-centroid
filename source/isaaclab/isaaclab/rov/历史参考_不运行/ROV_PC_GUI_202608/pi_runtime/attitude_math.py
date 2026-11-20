#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free 3-D IMU mounting and attitude-datum transformations."""

from __future__ import annotations

import math


IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def _finite(values, label):
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def normalize(quaternion):
    """Return a canonical unit quaternion in (w, x, y, z) order."""
    w, x, y, z = _finite(quaternion, "quaternion")
    magnitude = math.sqrt(w * w + x * x + y * y + z * z)
    if magnitude < 1e-9:
        raise ValueError("quaternion magnitude must be non-zero")
    result = (w / magnitude, x / magnitude, y / magnitude, z / magnitude)
    return result if result[0] >= 0.0 else tuple(-value for value in result)


def multiply(left, right):
    """Compose rotations: apply ``right`` first, then ``left``."""
    lw, lx, ly, lz = normalize(left)
    rw, rx, ry, rz = normalize(right)
    return normalize((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def inverse(quaternion):
    w, x, y, z = normalize(quaternion)
    return (w, -x, -y, -z)


def rotate_vector(quaternion, vector):
    """Rotate a 3-vector without treating it as a unit quaternion."""
    w, x, y, z = normalize(quaternion)
    vx, vy, vz = _finite(vector, "vector")
    # Unit-quaternion rotation matrix.  Keeping vector magnitude intact is
    # essential for angular rates; ``multiply`` intentionally normalizes and
    # therefore cannot be used for pure-vector quaternions.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def _dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cross(left, right):
    lx, ly, lz = _finite(left, "vector")
    rx, ry, rz = _finite(right, "vector")
    return (ly * rz - lz * ry, lz * rx - lx * rz, lx * ry - ly * rx)


def _unit(vector, label):
    value = _finite(vector, label)
    magnitude = math.sqrt(_dot(value, value))
    if magnitude < 1e-7:
        raise ValueError(f"{label} magnitude is too small")
    return tuple(component / magnitude for component in value)


def average_quaternions(quaternions):
    """Return the normalized chordal mean of nearby orientation samples.

    IMU samples that describe the same physical pose can use either quaternion
    sign.  Aligning every sample to the first one before summing prevents the
    antipodal representations from cancelling each other.  The capture window
    is deliberately short, so this dependency-free mean is appropriate for
    small sensor noise around one held pose.
    """
    samples = [normalize(value) for value in quaternions]
    if not samples:
        raise ValueError("quaternion samples are required")
    reference = samples[0]
    aligned = [
        sample if _dot(reference, sample) >= 0.0 else tuple(-v for v in sample)
        for sample in samples
    ]
    return normalize(tuple(sum(sample[index] for sample in aligned) for index in range(4)))


def quaternion_distance_deg(left, right):
    """Return the shortest 3-D angular distance between two orientations."""
    cosine = abs(_dot(normalize(left), normalize(right)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, cosine))))


def rotation_angle_deg(quaternion):
    """Return the canonical 0..180 degree angle represented by a quaternion."""
    w, x, y, z = normalize(quaternion)
    half_sine = math.sqrt(x * x + y * y + z * z)
    return math.degrees(2.0 * math.atan2(half_sine, w))


def rotation_axis(
    quaternion, *, minimum_angle_deg=8.0, maximum_angle_deg=None
):
    """Return the signed unit axis of a bounded captured rotation."""
    w, x, y, z = normalize(quaternion)
    half_sine = math.sqrt(x * x + y * y + z * z)
    angle_deg = rotation_angle_deg((w, x, y, z))
    if angle_deg < float(minimum_angle_deg):
        raise ValueError("captured_rotation_too_small")
    if maximum_angle_deg is not None and angle_deg > float(maximum_angle_deg):
        raise ValueError("captured_rotation_too_large")
    return _unit((x, y, z), "rotation axis")


def _symmetric_orthogonal_pair(
    first, second, *, minimum_separation_deg, label
):
    """Split measurement non-orthogonality equally between two captured axes."""
    first = _unit(first, f"{label} first axis")
    second = _unit(second, f"{label} second axis")
    cosine = max(-1.0, min(1.0, _dot(first, second)))
    separation = math.degrees(math.acos(cosine))
    separation_from_parallel = min(separation, 180.0 - separation)
    if separation_from_parallel < float(minimum_separation_deg):
        raise ValueError("captured_axes_too_close")

    # Unit vectors (a+b) and (a-b) are exactly orthogonal.  Recombining them
    # gives the closest equal-weight orthogonal pair without favouring either
    # operator capture.
    sum_axis = _unit(
        tuple(a + b for a, b in zip(first, second)), f"{label} axis sum"
    )
    difference_axis = _unit(
        tuple(a - b for a, b in zip(first, second)), f"{label} axis difference"
    )
    first_orthogonal = _unit(
        tuple(a + b for a, b in zip(sum_axis, difference_axis)),
        f"{label} orthogonal first axis",
    )
    second_orthogonal = _unit(
        tuple(a - b for a, b in zip(sum_axis, difference_axis)),
        f"{label} orthogonal second axis",
    )
    return first_orthogonal, second_orthogonal


def quaternion_from_basis_columns(x_axis, y_axis, z_axis):
    """Build a body->IMU quaternion from its three IMU-frame body axes."""
    x_axis = _unit(x_axis, "x axis")
    y_axis = _unit(y_axis, "y axis")
    z_axis = _unit(z_axis, "z axis")
    if abs(_dot(x_axis, y_axis)) > 1e-5 or abs(_dot(x_axis, z_axis)) > 1e-5 or abs(_dot(y_axis, z_axis)) > 1e-5:
        raise ValueError("basis axes must be orthogonal")
    # Columns are body X/Y/Z expressed in IMU coordinates.
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quat = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quat = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quat = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)
    return normalize(quat)


def mount_from_axis_captures(
    roll_delta,
    pitch_delta,
    *,
    minimum_angle_deg=8.0,
    maximum_angle_deg=None,
    minimum_separation_deg=60.0,
):
    """Infer the fixed body->IMU mount from pure roll and pitch captures.

    The deltas are baseline^-1 * captured raw orientation.  Their axes are
    ROV roll(X) and pitch(Y), expressed in IMU coordinates.  The third axis is
    formed by the right-handed cross product, so yaw is not guessed.
    """
    x_measured = rotation_axis(
        roll_delta,
        minimum_angle_deg=minimum_angle_deg,
        maximum_angle_deg=maximum_angle_deg,
    )
    y_measured = rotation_axis(
        pitch_delta,
        minimum_angle_deg=minimum_angle_deg,
        maximum_angle_deg=maximum_angle_deg,
    )
    x_axis, y_axis = _symmetric_orthogonal_pair(
        x_measured,
        y_measured,
        minimum_separation_deg=minimum_separation_deg,
        label="roll/pitch",
    )
    z_axis = _unit(_cross(x_axis, y_axis), "derived yaw axis")
    y_axis = _unit(_cross(z_axis, x_axis), "orthogonalized pitch axis")
    return quaternion_from_basis_columns(x_axis, y_axis, z_axis)


def mount_from_pitch_yaw_captures(
    pitch_delta,
    yaw_delta,
    *,
    minimum_angle_deg=20.0,
    maximum_angle_deg=45.0,
    minimum_separation_deg=60.0,
):
    """Infer the fixed body->IMU mount from positive pitch and yaw captures.

    The operator first holds the ROV at its level/forward reference, then makes
    a positive nose-up pitch rotation and, after returning level, a positive
    right/clockwise yaw rotation.  Their delta axes are body +Y and +Z expressed
    in the corrected IMU frame.  Body +X follows from Y cross Z, producing a
    right-handed basis even when the IMU itself is mounted diagonally.
    """
    pitch_measured = rotation_axis(
        pitch_delta,
        minimum_angle_deg=minimum_angle_deg,
        maximum_angle_deg=maximum_angle_deg,
    )
    yaw_measured = rotation_axis(
        yaw_delta,
        minimum_angle_deg=minimum_angle_deg,
        maximum_angle_deg=maximum_angle_deg,
    )
    y_axis, z_axis = _symmetric_orthogonal_pair(
        pitch_measured,
        yaw_measured,
        minimum_separation_deg=minimum_separation_deg,
        label="pitch/yaw",
    )
    x_axis = _unit(_cross(y_axis, z_axis), "derived roll axis")
    # Recompute Z from X cross Y to remove the final floating-point residue and
    # make the handedness explicit at the persistence boundary.
    z_axis = _unit(_cross(x_axis, y_axis), "orthogonalized yaw axis")
    return quaternion_from_basis_columns(x_axis, y_axis, z_axis)


def from_euler_deg(roll_deg, pitch_deg, yaw_deg):
    """Convert sensor roll(X), pitch(Y), yaw(Z) to a Z-Y-X quaternion."""
    roll, pitch, yaw = _finite((roll_deg, pitch_deg, yaw_deg), "Euler angles")
    roll *= math.pi / 360.0
    pitch *= math.pi / 360.0
    yaw *= math.pi / 360.0
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return normalize((
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ))


def _wrap_180(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def to_euler_deg(quaternion):
    """Return roll/yaw in [-180,180), pitch in its non-ambiguous ±90 range."""
    w, x, y, z = normalize(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, pitch_term)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return {
        "roll": _wrap_180(math.degrees(roll)),
        "pitch": math.degrees(pitch),
        "yaw": _wrap_180(math.degrees(yaw)),
    }


def raw_euler_to_vehicle_axes(raw_euler, axis_signs=(1.0, 1.0, 1.0)):
    """Convert a sensor report into the established right-handed ROV axes.

    The three signs are *sensor coordinate convention*, not balance trim.  A
    valid installation has an even number of reversed axes (determinant +1),
    for example the already bench-verified (+roll, -pitch, -yaw) convention.
    """
    roll_sign, pitch_sign, yaw_sign = _finite(axis_signs, "axis signs")
    signs = (roll_sign, pitch_sign, yaw_sign)
    if any(abs(abs(value) - 1.0) > 1e-9 for value in signs):
        raise ValueError("axis signs must each be +1 or -1")
    if roll_sign * pitch_sign * yaw_sign < 0.0:
        raise ValueError("axis signs must form a right-handed rotation")
    return from_euler_deg(
        roll_sign * raw_euler["roll"],
        pitch_sign * raw_euler["pitch"],
        yaw_sign * raw_euler["yaw"],
    )


def vehicle_body_rate_from_raw(
    raw_rate_dps,
    mount_quaternion=IDENTITY_QUATERNION,
    axis_signs=(1.0, 1.0, 1.0),
    *,
    mount_calibrated=False,
):
    """Express IMU gyroscope rates in calibrated ROV body axes.

    The sign triple first applies the already established right-handed sensor
    convention.  A calibrated mount quaternion stores body axes expressed in
    that corrected IMU frame, so its inverse maps the measured vector back to
    body axes.  The Options/reference quaternion is deliberately absent:
    angular velocity is a body vector and must not rotate with the chosen
    world datum.
    """
    return vehicle_vector_from_raw(
        raw_rate_dps,
        mount_quaternion,
        axis_signs,
        mount_calibrated=mount_calibrated,
        label="raw angular rate",
    )


def vehicle_vector_from_raw(
    raw_vector,
    mount_quaternion=IDENTITY_QUATERNION,
    axis_signs=(1.0, 1.0, 1.0),
    *,
    mount_calibrated=False,
    label="raw vector",
):
    """Express a sensor-frame vector in calibrated ROV body axes."""
    vector = _finite(raw_vector, label)
    signs = _finite(axis_signs, "axis signs")
    if any(abs(abs(value) - 1.0) > 1e-9 for value in signs):
        raise ValueError("axis signs must each be +1 or -1")
    if signs[0] * signs[1] * signs[2] < 0.0:
        raise ValueError("axis signs must form a right-handed rotation")
    signed = tuple(value * sign for value, sign in zip(vector, signs))
    if not mount_calibrated:
        return signed
    return rotate_vector(inverse(mount_quaternion), signed)


def tilt_compensated_magnetic_heading_deg(body_magnetic_field, roll_deg, pitch_deg):
    """Return raw magnetic course after removing ROV roll and pitch.

    The body-to-world attitude convention is Z-Y-X.  Applying the roll and
    pitch rotations to the body magnetic vector leaves yaw as the horizontal
    angle.  The returned sign matches the historical magnetometer course;
    installation-specific direction/offset remains a runtime configuration.
    """
    mx, my, mz = _finite(body_magnetic_field, "body magnetic field")
    roll, pitch = _finite((roll_deg, pitch_deg), "vehicle tilt")
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    horizontal_x = (
        mx * math.cos(pitch)
        + my * math.sin(roll) * math.sin(pitch)
        + mz * math.cos(roll) * math.sin(pitch)
    )
    horizontal_y = my * math.cos(roll) - mz * math.sin(roll)
    if math.hypot(horizontal_x, horizontal_y) < 1e-9:
        raise ValueError("horizontal magnetic field is too small")
    return math.degrees(math.atan2(horizontal_y, horizontal_x)) % 360.0


def attitude_from_raw_euler(
    raw_euler, mount_quaternion, reference_quaternion, axis_signs=(1.0, 1.0, 1.0)
):
    """Express raw IMU readings in the ROV frame and Options datum."""
    raw = raw_euler_to_vehicle_axes(raw_euler, axis_signs)
    vehicle = multiply(raw, mount_quaternion)
    referenced = multiply(inverse(reference_quaternion), vehicle)
    return to_euler_deg(referenced), vehicle
