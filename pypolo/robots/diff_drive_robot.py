import numpy as np
from typing import Tuple

from ..sensors import BaseSensor
from . import BaseRobot


class DiffDriveRobot(BaseRobot):

    def compute_action(self) -> Tuple[np.ndarray, float]:
        x, y, yaw = self.state
        goal_x, goal_y = self.goals[0][0], self.goals[0][1]
        x_diff, y_diff = goal_x - x, goal_y - y
        dist = np.hypot(x_diff, y_diff)
        x_odom = np.cos(yaw) * x_diff + np.sin(yaw) * y_diff
        y_odom = -np.sin(yaw) * x_diff + np.cos(yaw) * y_diff
        lin_vel = self.max_lin_vel * np.tanh(3 * x_odom)
        # lin_vel = np.clip(lin_vel, -self.max_lin_vel, self.max_lin_vel)
        ang_vel = np.arctan2(y_odom, x_odom)
        # ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel)
        return np.array([lin_vel, ang_vel]), dist

    def update_state(self, action) -> None:
        lin_vel, ang_vel = action
        self.state[0] += lin_vel * np.cos(self.state[2]) * self.control_dt
        self.state[1] += lin_vel * np.sin(self.state[2]) * self.control_dt
        self.state[2] += ang_vel * self.control_dt
        while self.state[2] > np.pi:
            self.state[2] -= 2 * np.pi
        while self.state[2] < -np.pi:
            self.state[2] += 2 * np.pi
            
    #For LidarSensor
    def step_with_heading(self, current_heading, num_targets) -> None:
        action, dist = self.compute_action()
        if dist < self.goal_radius:
            self.goals = self.goals[1:]
        self.update_state(action)
        self.timer += self.control_dt
        if self.timer > self.sensor.dt:
            
            location, observations = self.sensor.sense(self.state, current_heading, num_targets=num_targets)
            self.sampled_locations.append(location)
            self.sampled_observations.append(observations)
            self.timer = 0.0

