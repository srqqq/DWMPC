from typing import NamedTuple

import numpy as np


class DisturbanceSample(NamedTuple):
    active: bool
    elapsed_time: float
    force_reference_heading: np.ndarray
    force_world: np.ndarray


def sample_sinusoidal_lateral_force(
    trajectory_time,
    reference_yaw,
    robot_weight,
    *,
    enabled,
    start_delay,
    amplitude_ratio,
    frequency,
    phase,
    ramp_cycles,
    steady_cycles,
):
    """Sample a force along the reference-heading frame's lateral axis."""
    if frequency <= 0.0 or robot_weight <= 0.0:
        raise ValueError("disturbance frequency and robot weight must be positive")
    if start_delay < 0.0 or amplitude_ratio < 0.0:
        raise ValueError("disturbance start delay and amplitude ratio must be non-negative")
    if ramp_cycles < 0.0 or steady_cycles < 0.0:
        raise ValueError("disturbance cycle counts must be non-negative")

    elapsed_time = float(trajectory_time - start_delay)
    ramp_duration = ramp_cycles / frequency
    duration = (2.0 * ramp_cycles + steady_cycles) / frequency
    time_tolerance = 1e-12
    active = bool(
        enabled
        and elapsed_time >= -time_tolerance
        and elapsed_time < duration - time_tolerance
    )

    force_reference_heading = np.zeros(3)
    force_world = np.zeros(3)
    if not active:
        return DisturbanceSample(
            active, elapsed_time, force_reference_heading, force_world
        )

    elapsed_time = max(elapsed_time, 0.0)

    envelope = 1.0
    if ramp_duration > 0.0:
        if elapsed_time < ramp_duration:
            envelope = np.sin(0.5 * np.pi * elapsed_time / ramp_duration) ** 2
        elif duration - elapsed_time < ramp_duration:
            envelope = np.sin(
                0.5 * np.pi * (duration - elapsed_time) / ramp_duration
            ) ** 2

    lateral_force = (
        amplitude_ratio
        * robot_weight
        * envelope
        * np.sin(2.0 * np.pi * frequency * elapsed_time + phase)
    )
    force_reference_heading[1] = lateral_force
    force_world[:2] = lateral_force * np.array([
        -np.sin(reference_yaw),
        np.cos(reference_yaw),
    ])
    return DisturbanceSample(
        active, elapsed_time, force_reference_heading, force_world
    )
