from enum import Enum

class LaneDirection(Enum):
    UNSPECIFIED = 0
    START_TO_END = 1
    END_TO_START = 2


class RelativePositionInLane(Enum):
    START = 0,
    END = 1,
    MIDDLE = 2


class Lane:

    def __init__(self, line1, line2, direction=LaneDirection.START_TO_END):
        self.line1 = line1
        self.line2 = line2
        self.direction = direction
        self.bottom_left = None
        self.top_right = None
        self.center = None

        self.compute_attributes()

    def compute_attributes(self):
        self.bottom_left = min(self.line1.start[0], self.line1.start[0]), min(self.line1.start[1], self.line2.start[1])
        self.top_right = max(self.line1.end[0], self.line1.end[0]), max(self.line1.end[1], self.line2.end[1])
        self.center = (self.line1.start[0] + self.line1.end[0]) / 2, (self.line1.end[1] + self.line2.end[1]) / 2

    def start_middle(self):
        if self.direction == LaneDirection.START_TO_END:
            return self.bottom_left[0], self.center[1]
        return self.top_right[0], self.center[1]

    def end_middle(self):
        if self.direction == LaneDirection.START_TO_END:
            return self.top_right[0], self.center[1]
        return self.bottom_left[0], self.center[1]


