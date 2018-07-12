import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point

from .....semantics import Line
from .scenario import Scenario


class TIntersectionScenario(Scenario):

    def __init__(self, x, y, lane_length, lane_width, boundaries):
        Scenario.__init__(self, x, y, lane_length, lane_width, boundaries)

        self.polygon = np.array([
            (self.x, self.y),
            (self.x + lane_length, self.y),
            (self.x + lane_length, max(self.y - lane_length, 0)),
            (self.x, self.y + lane_width),
            (self.x + lane_length, self.y + lane_width),
            (self.x + lane_length, min(self.y + lane_width + lane_length, boundaries[1])),
            (self.x + lane_length + lane_width, min(self.y + lane_width + lane_length, boundaries[1])),
            (self.x + lane_length + lane_width,  self.y + lane_width),
            (self.x + lane_length + lane_width,  self.y + lane_width),
            (self.x + lane_length + lane_width, max(self.y - lane_length, 0))
        ])

        self.lines = [
            Line(self.polygon[0], self.polygon[1]),
            Line(self.polygon[1], self.polygon[2]),
            Line(self.polygon[3], self.polygon[4]),
            Line(self.polygon[4], self.polygon[5]),
            Line(self.polygon[6], self.polygon[7]),
            Line(self.polygon[7], self.polygon[8]),
            Line(self.polygon[8], self.polygon[9])
        ]

        self.trajectories = np.array([
            [
                (self.x, self.y + lane_width / 2),
                (self.x + lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length / 2, self.y + lane_width / 2),
                (self.x + 3 * lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length, self.y + lane_width / 2),
                (self.x + lane_length + lane_width / 2, self.y + lane_width / 2),
                (self.x + lane_length + lane_width / 2, self.polygon[1][1]),
                (self.x + lane_length + lane_width / 2, (self.polygon[2][1] + self.polygon[1][1]) / 2),
                (self.x + lane_length + lane_width / 2, self.polygon[2][1])
            ],
            [
                (self.x, self.y + lane_width / 2),
                (self.x + lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length / 2, self.y + lane_width / 2),
                (self.x + 3 * lane_length / 4, self.y + lane_width / 2),
                (self.x + lane_length, self.y + lane_width / 2),
                (self.x + lane_length + lane_width / 2, self.y + lane_width / 2),
                (self.x + lane_length + lane_width / 2, self.polygon[4][1]),
                (self.x + lane_length + lane_width / 2, (self.polygon[4][1] + self.polygon[5][1]) / 2),
                (self.x + lane_length + lane_width / 2, self.polygon[5][1])
            ],
        ])

        self.shape = MultiPolygon([
            Polygon(self.polygon[[0, 1, 4, 3]]),  # lane # 1
            Polygon(self.polygon[[1, 2, 9, 8]]),
            Polygon(self.polygon[[1, 8, 7, 4]]),
            Polygon(self.polygon[[4, 7, 6, 5]]),
        ])