from math import cos, sin, sqrt

import numpy as np
from .pycubicspline import Spline2D
from ....simulator import Agent


def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_closest_point(trajectory, point):
    return np.argmin(np.array([euclidean_distance(point, p1) for p1 in trajectory]))


class TeacherPurePursuit(Agent):
    def learn(self, state, action):
        raise NotImplementedError()

    def explore(self, state, horizon=1):
        raise NotImplementedError()

    def __init__(self, world):
        Agent.__init__(self, world)

    def plan(self, trajectory, horizon=1):
        path = []
        closest_point_index = calculate_closest_point(trajectory, (self.x, self.y))

        distance = 0.0
        look_ahead = 1.0
        # look ahead
        while distance < look_ahead and closest_point_index < len(trajectory):
            if closest_point_index + 1 < len(trajectory):
                distance += euclidean_distance(trajectory[closest_point_index], trajectory[closest_point_index + 1])
            closest_point_index += 1

        if closest_point_index < len(trajectory):
            closest_trajectory = trajectory[closest_point_index:]
            sample_x = np.insert(closest_trajectory[::, 0], 0, self.x)
            sample_y = np.insert(closest_trajectory[::, 1], 0, self.y)

            spline = Spline2D(sample_x, sample_y)
            spline_samples = np.arange(0, spline.s[-1], self.v)
            for samples in spline_samples:
                x, y = spline.calc_position(samples)
                theta = spline.calc_yaw(samples)
                path.append((x, y, theta))
        else:
            path.append((self.x, self.y, self.theta))

        if horizon:
            return np.array(path)[:horizon]
        return np.array(path)

    def exploit(self, state, horizon=1):
        raise NotImplementedError()

    def execute(self, action):
        self.theta = action
        self.x = self.x + self.v * cos(self.theta)
        self.y = self.y + self.v * sin(self.theta)
