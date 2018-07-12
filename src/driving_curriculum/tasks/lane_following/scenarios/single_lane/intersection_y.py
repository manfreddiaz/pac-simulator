from math import sqrt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point

from .scenario import Scenario


class YIntersectionScenario(Scenario):

    def __init__(self, x, y, lane_length, lane_width, boundaries):
        Scenario.__init__(self, x, y, lane_length, lane_width, boundaries)
        c1 = sqrt(3) / 2 * lane_width
        c2 = sqrt(lane_length ** 2 / 4 - 3 / 4 * lane_width ** 2)

        self.polygon = np.array([
            (self.x, self.y),
            (self.x + lane_length, self.y),
            (self.x + lane_length + c1, self.y - c2),
            (self.x, self.y + lane_width),
            (self.x + lane_length, self.y + lane_width),
            (self.x + lane_length + c1, self.y + lane_width + c2),
            (self.x + lane_length + 2 * c1, self.y + lane_width / 2 - c2),
            (self.x + lane_length + c1, self.y + lane_width / 2),
            (self.x + lane_length + 2 * c1, self.y + lane_width / 2 + c2)
        ])

        self.lines = np.array([
            (self.polygon[0], self.polygon[1]),
            (self.polygon[1], self.polygon[2]),
            (self.polygon[3], self.polygon[4]),
            (self.polygon[4], self.polygon[5]),
            (self.polygon[6], self.polygon[7]),
            (self.polygon[7], self.polygon[8]),
        ])

        self.trajectories = np.array([
            [
                (self.x, self.y + lane_width / 2),
                (self.x + lane_length / 4,
                 self.y + lane_width / 2),
                (self.x + lane_length / 2, self.y + lane_width / 2),
                (self.x + 3 * lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length, self.y + lane_width / 2),
                ((self.polygon[4][0] + self.polygon[7][0]) / 2, (self.polygon[4][1] + self.polygon[7][1]) / 2),
                ((3 * self.polygon[4][0] + 3 * self.polygon[7][0] + self.polygon[5][0] + self.polygon[8][0]) / 8,
                    (3 * self.polygon[4][1] + 3 * self.polygon[7][1] + self.polygon[5][1] + self.polygon[8][1]) / 8),
                ((self.polygon[4][0] + self.polygon[7][0] + self.polygon[5][0] + self.polygon[8][0]) / 4,
                    (self.polygon[4][1] + self.polygon[7][1] + self.polygon[5][1] + self.polygon[8][1]) / 4),
                ((self.polygon[4][0] + self.polygon[7][0] + 3 * self.polygon[5][0] + 3 * self.polygon[8][0]) / 8,
                    (self.polygon[4][1] + self.polygon[7][1] + 3 * self.polygon[5][1] + 3 * self.polygon[8][1]) / 8),
                ((self.polygon[5][0] + self.polygon[8][0]) / 2, self.polygon[5][1])
            ],
            [
                (self.x, self.y + lane_width / 2),
                (self.x + lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length / 2, self.y + lane_width / 2),
                (self.x + 3 * lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length, self.y + lane_width / 2),
                ((self.polygon[1][0] + self.polygon[7][0]) / 2, (self.polygon[1][1] + self.polygon[7][1]) / 2),
                ((3 * self.polygon[1][0] + 3 * self.polygon[7][0] + self.polygon[2][0] + self.polygon[6][0]) / 8,
                 (3 * self.polygon[1][1] + 3 * self.polygon[7][1] + self.polygon[2][1] + self.polygon[6][1]) / 8),
                ((self.polygon[1][0] + self.polygon[7][0] + self.polygon[2][0] + self.polygon[6][0]) / 4,
                 (self.polygon[1][1] + self.polygon[7][1] + self.polygon[2][1] + self.polygon[6][1]) / 4),
                ((self.polygon[1][0] + self.polygon[7][0] + 3 * self.polygon[2][0] + 3 * self.polygon[6][0]) / 8,
                 (self.polygon[1][1] + self.polygon[7][1] + 3 * self.polygon[2][1] + 3 * self.polygon[6][1]) / 8),
                ((self.polygon[2][0] + self.polygon[6][0]) / 2, self.polygon[2][1])
            ]
        ])
        self.shape = MultiPolygon([
            Polygon(self.polygon[[0, 1, 4, 3]]),  # lane # 1
            Polygon(self.polygon[[1, 2, 6, 7]]),
            Polygon(self.polygon[[1, 7, 4]]),  # exchange zone
            Polygon(self.polygon[[7, 4, 5, 8]]),
        ])
