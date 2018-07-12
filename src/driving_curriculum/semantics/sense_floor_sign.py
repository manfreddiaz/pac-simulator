from .lane import RelativePositionInLane, LaneDirection


class SenseFloorSign:
    def __init__(self, lane, position=RelativePositionInLane.MIDDLE, color=(1.0, 1.0, 1.0)):
        self.lane = lane
        self.position = position
        self.center = self.compute_center()
        self.direction = self.compute_direction()
        self.color = color

    def compute_center(self):
        if self.position == RelativePositionInLane.MIDDLE:
            x, y = self.lane.center
        elif self.position == RelativePositionInLane.START:
            x, y = self.lane.start_middle()
        else:
            x, y = self.lane.end_middle()
        return x, y

    def compute_direction(self):
        sense = 1
        if self.lane.direction == LaneDirection.END_TO_START:
            sense = -1
        return sense
