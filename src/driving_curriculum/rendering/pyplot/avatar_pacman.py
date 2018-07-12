from math import degrees, pi
import matplotlib.patches as patch


from ....simulator.graphics import Visual


class PacmanAvatarPyplot(Visual):
    MOUTH_ANGLE = pi / 4

    def __init__(self, world, radius, color):
        self.world = world
        self.radius = radius
        self.color = color
        self.graphics = None

    def render(self):
        self.graphics = patch.Wedge(center=(0, 0), r=self.radius, color=self.color,
                                    theta1=degrees(PacmanAvatarPyplot.MOUTH_ANGLE),
                                    theta2=degrees(-PacmanAvatarPyplot.MOUTH_ANGLE))

    def update(self):
        self.graphics.set(center=(self.world.active_agent.x, self.world.active_agent.y),
                          theta1=degrees(PacmanAvatarPyplot.MOUTH_ANGLE + self.world.active_agent.theta),
                          theta2=degrees(-PacmanAvatarPyplot.MOUTH_ANGLE + self.world.active_agent.theta))
