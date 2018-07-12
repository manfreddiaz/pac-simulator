from math import sqrt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point

from .scenario import Scenario
np.random.seed(1234)


def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


class HighwayExitsScenario(Scenario):
    def __init__(self, x, y, lane_length, lane_width, boundaries):
        Scenario.__init__(self, x, y, lane_length, lane_width, boundaries)
        c1 = sqrt(lane_length ** 2 - lane_width ** 2)
        
        self.polygon = np.array([
            (self.x, self.y),
            (self.x + lane_length, self.y),
            (self.x + lane_length + lane_width, self.y - c1),
            (self.x, self.y + lane_width),
            (self.x + lane_length, self.y + lane_width),
            (self.x + lane_length + lane_width, self.y + lane_width + c1),
            (self.x + lane_length + 2 * lane_width, self.y - c1),
            (self.x + lane_length + lane_width, self.y),
            (boundaries[0], self.y),
            (boundaries[0], self.y + lane_width),
            (self.x + lane_length + lane_width, self.y + lane_width),
            (self.x + lane_length + 2 * lane_width, self.y + lane_width + c1)
        ])
       
        self.lines = np.array([
            # first lane (down & up)
            (self.polygon[0], self.polygon[1]),
            (self.polygon[3], self.polygon[4]),
            # first exit (top)
            (self.polygon[4], self.polygon[5]),
            (self.polygon[10], self.polygon[11]),
            # second exit (bottom)
            (self.polygon[1], self.polygon[2]),
            (self.polygon[6], self.polygon[7]),
            # continuation lane
            (self.polygon[7], self.polygon[8]),
            (self.polygon[10], self.polygon[9]),
        ])

        self.trajectories = np.array([
            [
                midpoint(self.polygon[0], self.polygon[3]),
                midpoint(midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])), midpoint(self.polygon[0], self.polygon[3])),
                midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])),
                midpoint(midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])), midpoint(self.polygon[1], self.polygon[4])),
                midpoint(self.polygon[1], self.polygon[4]),
                midpoint(self.polygon[4], self.polygon[10]),
                midpoint(midpoint(midpoint(self.polygon[4], self.polygon[10]), midpoint(self.polygon[5], self.polygon[11])), midpoint(self.polygon[4], self.polygon[10])),
                midpoint(midpoint(self.polygon[4], self.polygon[10]), midpoint(self.polygon[5], self.polygon[11])),
            ],
            [
                midpoint(self.polygon[0], self.polygon[3]),
                midpoint(midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])), midpoint(self.polygon[0], self.polygon[3])),
                midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])),
                midpoint(midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])), midpoint(self.polygon[1], self.polygon[4])),
                midpoint(self.polygon[1], self.polygon[4]),
                midpoint(self.polygon[1], self.polygon[7]),
                midpoint(midpoint(midpoint(self.polygon[1], self.polygon[7]), midpoint(self.polygon[2], self.polygon[6])), midpoint(self.polygon[1], self.polygon[7])),
                midpoint(midpoint(self.polygon[1], self.polygon[7]), midpoint(self.polygon[2], self.polygon[6])),
            ],
            [
                midpoint(self.polygon[0], self.polygon[3]),
                midpoint(midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])), midpoint(self.polygon[0], self.polygon[3])),
                midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])),
                midpoint(midpoint(midpoint(self.polygon[0], self.polygon[3]), midpoint(self.polygon[1], self.polygon[4])), midpoint(self.polygon[1], self.polygon[4])),
                midpoint(self.polygon[1], self.polygon[4]),
                midpoint(self.polygon[7], self.polygon[10]),
                midpoint(midpoint(self.polygon[8], self.polygon[9]), midpoint(self.polygon[7], self.polygon[10])),
                midpoint(self.polygon[8], self.polygon[9]),
            ]
        ])

        self.shape = MultiPolygon([
            Polygon(self.polygon[[0, 1, 4, 3]]),  # lane # 1
            Polygon(self.polygon[[1, 2, 6, 7]]),
            Polygon(self.polygon[[1, 7, 10, 4]]),  # exchange zone
            Polygon(self.polygon[[7, 8, 9, 10]]),
            Polygon(self.polygon[[4, 5, 11, 10]]),
        ])
