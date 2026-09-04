import numpy as np


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rotation(yaw):
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]])


class CircleTrajectoryPlanner:
    def __init__(self, x0, y0, yaw0, radius, speed):
        if radius <= 0.0 or speed <= 0.0:
            raise ValueError("circle radius and speed must be positive")
        self.origin = np.array([x0, y0], dtype=float)
        self.rotation = _rotation(yaw0)
        self.yaw0 = yaw0
        self.radius = radius
        self.speed = speed
        self.angular_speed = speed / radius

    def get_plan_state(self, time):
        angle = self.angular_speed * time
        position_local = self.radius * np.array([
            np.sin(angle),
            1.0 - np.cos(angle),
        ])
        velocity_local = self.speed * np.array([
            np.cos(angle),
            np.sin(angle),
        ])
        position = self.origin + self.rotation @ position_local
        velocity = self.rotation @ velocity_local
        yaw = normalize_angle(self.yaw0 + angle)
        return np.array([
            position[0], position[1], yaw,
            velocity[0], velocity[1], self.angular_speed,
        ])


class FigureEightTrajectoryPlanner:
    """Gerono figure-eight trajectory with an initial tangent aligned to yaw0."""

    def __init__(self, x0, y0, yaw0, x_amplitude, y_amplitude, period):
        if x_amplitude <= 0.0 or y_amplitude <= 0.0 or period <= 0.0:
            raise ValueError("figure-eight amplitudes and period must be positive")
        self.origin = np.array([x0, y0], dtype=float)
        self.x_amplitude = x_amplitude
        self.y_amplitude = y_amplitude
        self.angular_speed = 2.0 * np.pi / period

        initial_tangent_angle = np.arctan2(2.0 * y_amplitude, x_amplitude)
        self.rotation = _rotation(yaw0 - initial_tangent_angle)

    def get_plan_state(self, time):
        angle = self.angular_speed * time
        position_local = np.array([
            self.x_amplitude * np.sin(angle),
            self.y_amplitude * np.sin(2.0 * angle),
        ])
        velocity_local = self.angular_speed * np.array([
            self.x_amplitude * np.cos(angle),
            2.0 * self.y_amplitude * np.cos(2.0 * angle),
        ])
        acceleration_local = self.angular_speed**2 * np.array([
            -self.x_amplitude * np.sin(angle),
            -4.0 * self.y_amplitude * np.sin(2.0 * angle),
        ])

        position = self.origin + self.rotation @ position_local
        velocity = self.rotation @ velocity_local
        speed_squared = velocity_local @ velocity_local
        yaw_rate = (
            velocity_local[0] * acceleration_local[1]
            - velocity_local[1] * acceleration_local[0]
        ) / speed_squared
        yaw = np.arctan2(velocity[1], velocity[0])
        return np.array([
            position[0], position[1], yaw,
            velocity[0], velocity[1], yaw_rate,
        ])


def create_trajectory_planner(mode, x0, y0, yaw0, parameters):
    if mode == "circle":
        return CircleTrajectoryPlanner(x0, y0, yaw0, **parameters)
    if mode == "figure_eight":
        return FigureEightTrajectoryPlanner(x0, y0, yaw0, **parameters)
    raise ValueError(f"unsupported automatic trajectory mode: {mode}")
