from math import sin, atan2, asin, acos, pi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from curriculum_learning.learning.end_to_end.online import OnlineLaneFollowing

np.random.seed(1234)


class GoalOrientedOnlineLaneFollowing(OnlineLaneFollowing):

    def __init__(self, world, teacher, learner, scenario, storage_location):
        OnlineLaneFollowing.__init__(self, world, teacher, learner, scenario, storage_location)
        # last_points = [trajectory[-1] for trajectory in self.scenario.trajectories]
        # self.goals = [(point[0] / world.width, point[1] / world.height) for point in last_points]  # normalized
        self.goals = [sin(pi/2), sin(-pi/2)]

    def learn(self, trajectory_index, alpha):
        training = True
        while training:
            observation = self.world.observe()
            learn = np.random.choice([True, False], p=[(1 - alpha), alpha])
            if learn:
                self.embody_learner()
                query = self.teacher.plan(self.scenario.trajectories[trajectory_index])
                x, y, angle = query[0]
                loss = self.learner.learn(observation, np.array([2 * angle / pi]), self.goals[trajectory_index])
                self.world.active_agent = self.learner
                self.learner.execute(angle)
                self.progress_bar.set_postfix({'loss': loss, 'alpha': alpha, 'learning': learn})
            else:  # exploit
                self.world.active_agent = self.learner
                prediction = self.learner.exploit(observation, self.goals[trajectory_index])
                self.learner.execute(prediction[0] * pi / 2)
                self.progress_bar.set_postfix({'alpha': alpha, 'learning': learn})

            training = not self.scenario.in_trajectory_goal(self.learner.x, self.learner.y) \
                       and self.scenario.in_scenario(self.learner.x, self.learner.y)

        return self.scenario.in_trajectory_goal(self.learner.x, self.learner.y)

    def test(self):
        self.world.active_agent = self.learner
        self.world.render()

        observation = self.world.observe()
        self.learner.test(observation.shape, (1, ), self.storage_location)

        figure = plt.figure(2)
        axis = figure.subplots(2, 3)

        samples = self.scenario.samples(300, seed=4567)
        i = 0
        self.progress_bar = tqdm(samples)
        for sample in self.progress_bar:
            self.learner.x = sample[0]
            self.learner.y = sample[1]
            self.learner.theta = sample[2]
            self.learner.v = 1.0
            while self.scenario.in_scenario(self.learner.x, self.learner.y) and self.is_agent_in_world_bound():
                mean, aleatoric, epistemic, mixture, _ = self.learner.exploit(self.world.observe(), self.goals[0])
                self.learner.execute(mean * pi / 2)
                axis[0][0].scatter(i, asin(aleatoric))
                axis[0][1].scatter(i, asin(epistemic))
                axis[0][2].scatter(i, mixture)
                # axis[1][0].scatter(i, acos(aleatoric[1]) - pi/2)
                # axis[1][1].scatter(i, acos(epistemic[1]) - pi/2)
                # axis[1][2].scatter(i, mixture[1])

                i = i + 1

    def demo_learner(self):
        figure = plt.figure(2)
        axis = figure.subplots(2, 3)

        self.world.render()
        self.world.active_agent = self.learner
        i = 0
        in_scenario = True
        while in_scenario:
            observation = self.world.observe()
            sin_0, cos_0 = self.learner.exploit(observation)
            self.learner.execute(atan2(sin_0[0], cos_0[0]))
            in_scenario = self.is_agent_in_world_bound()  # self.scenario.in_scenario(self.learner.x, self.learner.y)
            # debug
            axis[0][0].scatter(i, asin(sin_0[1]))
            # axis[0][1].scatter(i, asin(sin_0[2]))
            axis[0][2].scatter(i, sin_0[3])
            axis[1][0].scatter(i, acos(cos_0[1]))
            # axis[1][1].scatter(i, acos(cos_0[2]))
            axis[1][2].scatter(i, cos_0[3])

            i = i + 1
        plt.pause(0.01)

    def demo_teacher(self):
        self.world.active_agent = self.teacher
        self.world.render()

        plt.scatter(self.scenario.trajectories[0][::, 0], self.scenario.trajectories[0][::, 1])

        samples = self.scenario.samples(300)
        self.progress_bar = tqdm(samples)
        for sample in self.progress_bar:
            for index in range(len(self.scenario.trajectories)):
                self.teacher.x = sample[0]
                self.teacher.y = sample[1]
                self.teacher.theta = sample[2]
                self.teacher.v = 1.0
                while self.scenario.in_scenario(self.teacher.x, self.teacher.y):
                    query = self.teacher.plan(self.scenario.trajectories[index], horizon=1)
                    # plt.scatter(query[::, 0], query[::, 1])
                    for instruction in query:
                        self.teacher.execute(instruction[2])
                        self.world.update()

    def is_agent_in_world_bound(self):
        agent = self.world.active_agent
        if 0 <= agent.x <= self.world.width \
                and 0 <= agent.y <= self.world.height:
            return True
        return False
