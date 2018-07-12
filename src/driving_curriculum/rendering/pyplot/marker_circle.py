import matplotlib.patches as patches

from ....simulator.graphics import Visual


class CirclePyplot(Visual):
    def __init__(self, center, radius=1.0, width=1.0, color=(1.0, 1.0, 1.0),
                 facecolor=(0.0, 0.0, 0.0), discontinuous=False):
        self.center = center
        self.radius = radius
        self.graphics = patches.Circle(xy=center, radius=radius, linewidth=width, edgecolor=color,
                                       facecolor=facecolor, zorder=0)

        if discontinuous:
            self.graphics.set_linestyle('--')
