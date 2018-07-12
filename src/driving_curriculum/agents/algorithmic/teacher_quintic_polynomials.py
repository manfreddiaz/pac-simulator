from math import cos, sin
import numpy as np

from ....simulator import Agent
from .quintic_polynomials_planner import quinic_polynomials_planner


class TeacherQuinticPolynomials(Agent):
    def learn(self, state, action):
        raise NotImplementedError()

    def explore(self, state, horizon=1):
        raise NotImplementedError()

    def __init__(self, world, lane):
        Agent.__init__(self, world)
        self.lane = lane
        self.navigation_plan = None
        self.goal = self.lane.end_middle()
        self.goal = self.goal[0], self.goal[1], 0.0  # the angle depends on the lane direction

    def plan(self, horizon=10):
        trajectory = quinic_polynomials_planner(sx=self.x, sy=self.y, syaw=self.theta, sv=self.v, sa=0.0,
                                          gx=self.goal[0], gy=self.goal[1], gyaw=self.goal[2], gv=0.0, ga=0.0,
                                          max_accel=0.0, max_jerk=0.1, dt=1)
        return np.array(trajectory[3])[:horizon]

    def exploit(self, state, horizon=1):
        if self.navigation_plan is None:
            self.navigation_plan = self.plan()
        for _ in range(horizon):
            self.execute()

    def execute(self, action, horizon=1):
        for _ in range(horizon):
            self.x = self.x + self.v * cos(action)
            self.y = self.y + self.v * sin(action)
