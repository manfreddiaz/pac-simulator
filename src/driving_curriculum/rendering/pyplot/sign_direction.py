import matplotlib.patches as patch

from ....simulator.graphics import Visual


class DirectionSignPyplot(Visual):
    def __init__(self, floor_sign, scale=5, width=15):
        self.floor_sign = floor_sign
        self.graphics = patch.Arrow(x=floor_sign.center[0] - floor_sign.direction * width / 2, y=floor_sign.center[1],
                                    dx=floor_sign.direction * width, dy=0.0, width=scale, color=floor_sign.color, zorder=0)

