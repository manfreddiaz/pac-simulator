from math import degrees, pi
import matplotlib.patches as patch

from ....simulator.graphics import Visual


class RectangularAvatarPyplot(Visual):
    def __init__(self, agent, height, width, color):
        self.agent = agent
        self.height = height
        self.width = width
        self.color = color
        self.graphics = patch.Rectangle(xy=[agent.x - width / 2, agent.y - height / 2],
                                        width=width, height=height, angle=agent.theta, color=color)

    def update(self):
        self.graphics.angle = degrees(self.agent.theta)
        self.graphics.set(xy=(self.agent.x - self.width / 2, self.agent.y - self.height / 2))
