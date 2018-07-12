from simulator.world import World
from simulator.agent import Agent


class ImitationLearningRegime:
    def __init__(self, world, teacher, learner):
        self.world = world
        self.teacher = teacher
        self.learner = learner

    def learn(self):
        raise NotImplementedError()

    def explore(self):
        raise NotImplementedError()

