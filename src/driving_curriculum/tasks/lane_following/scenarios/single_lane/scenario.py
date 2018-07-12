import numpy as np
from shapely.geometry import Point

np.random.seed(1234)


class Scenario:
    def __init__(self, x, y, lane_length, lane_width, boundaries):
        self.x = x
        self.y = y
        self.lane_length = lane_length
        self.lane_width = lane_width
        self.world_bound = boundaries
        self.polygon = None
        self.lines = None
        self.trajectories = None
        self.shape = None

    def samples(self, num_samples, seed=None):
        if seed:
            np.random.seed(seed)

        min_x, min_y, max_x, max_y = self.shape.geoms[0].bounds
        samples = []

        while len(samples) < num_samples:
            x, y, theta = (
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y),
                np.random.uniform(-np.pi / 2, np.pi / 2)
            )
            if self.shape.contains(Point(x, y)):
                samples.append((x, y, theta))

        return np.array(samples)

    def in_scenario(self, x, y):
        return self.shape.contains(Point(x, y))

    def in_trajectory_goal(self, x, y, radio=1.0):
        in_trajectory = False
        for index in range(len(self.trajectories)):
            final_x, final_y = self.trajectories[index][-1]
            in_trajectory = in_trajectory or (final_x - x) ** 2 + (final_y - y) ** 2 < radio ** 2
        return in_trajectory
