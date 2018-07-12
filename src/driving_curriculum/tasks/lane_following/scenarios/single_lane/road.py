import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon

from .scenario import Scenario


class RoadScenario(Scenario):

    def __init__(self, x, y, lane_length, lane_width, boundaries: tuple):
        Scenario.__init__(self, x, y, lane_length, lane_width, boundaries)

        self.polygon = np.array([
            (x, y),
            (min(boundaries[0], self.x + 2 * lane_length), self.y),
            (min(boundaries[0], self.x + 2 * lane_length), self.y + lane_width),
            (self.x, self.y + lane_width),
        ])
        self.trajectories = np.array([
            [
                (self.x, self.y + lane_width / 2),
                (self.x + lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length / 2, self.y + lane_width / 2),
                (self.x + 3 * lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length, self.y + lane_width / 2),
                (self.x + 5 * lane_length / 4, self.y + lane_width / 2),
                (self.x + 3 * lane_length / 2, self.y + lane_width / 2),
                (self.x + 7 * lane_length / 4, self.y + lane_width / 2),
                (self.x + 2 * lane_length, self.y + lane_width / 2),
            ]
        ])
        self.lines = np.array([
            (self.polygon[0], self.polygon[1]),
            (self.polygon[2], self.polygon[3])
        ])
        self.shape = MultiPolygon([
            Polygon(self.polygon)
        ])
