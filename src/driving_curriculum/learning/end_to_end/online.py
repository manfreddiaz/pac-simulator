from math import atan2, asin, acos, pi, sqrt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from simulator.learning import OnlineImitationLearning

np.random.seed(1234)

SUCCESSFUL_ATTEMPTS = 'SUCCESSFUL_ATTEMPTS'
UNSUCCESSFUL_ATTEMPTS = 'UNSUCCESSFUL_ATTEMPTS'
PATIENCE_TRESHOLD = 1
N_SAMPLES = 400


class OnlineLaneFollowing(OnlineImitationLearning):

    def __init__(self, world, teacher, learner, scenario, storage_location):
        OnlineImitationLearning.__init__(self, world, teacher, learner)
        self.storage_location = storage_location
        self.learning_statistics = {
            SUCCESSFUL_ATTEMPTS: [],
            UNSUCCESSFUL_ATTEMPTS: []
        }
        self.scenario = scenario
        self.progress_bar = None

    def train(self):
        self.world.active_agent = self.learner
        self.world.render()
        observation = self.world.observe()
        self.learner.train(observation.shape, (1,), )

        samples = self.scenario.samples(400)
        self.progress_bar = tqdm(samples)
        for sample_index, sample in enumerate(self.progress_bar):
            for trajectory_index in range(len(self.scenario.trajectories)):
                alpha = (sample_index + 1) / float(N_SAMPLES)
                finished = False
                patience = 0
                while not finished:
                    self.learner.x = sample[0]
                    self.learner.y = sample[1]
                    self.learner.theta = sample[2]
                    self.learner.v = 1.0
                    finished = self.learn(trajectory_index, alpha)
                    patience += 1
                # print(patience)
                self.learner.commit()

                if finished:
                    self.learning_statistics[SUCCESSFUL_ATTEMPTS].append((sample_index, trajectory_index))
                else:
                    self.learning_statistics[UNSUCCESSFUL_ATTEMPTS].append((sample_index, trajectory_index))

        self.debug_statistics()

    def debug_statistics(self):
        trajectory_success = []
        trajectory_unsuccess = []
        for _ in enumerate(self.scenario.trajectories):
            trajectory_success.extend([0])
            trajectory_unsuccess.extend([0])

        for attempt in self.learning_statistics[SUCCESSFUL_ATTEMPTS]:
            trajectory_success[attempt[1]] += 1

        for attempt in self.learning_statistics[UNSUCCESSFUL_ATTEMPTS]:
            trajectory_unsuccess[attempt[1]] += 1

        print(trajectory_success)
        print(trajectory_unsuccess)

    def closest_goal(self):
        min_dist = 10000000
        min_index = -1
        for index in range(len(self.scenario.trajectories)):
            gx, gy = self.scenario.trajectories[index][-1]
            distance = sqrt((gx - self.learner.x) ** 2 + (gy - self.learner.y) ** 2)
            if distance < min_dist:
                min_dist = distance
                min_index = index
        return min_index

    def learn(self, trajectory_index, alpha):
        training = True
        while training:
            observation = self.world.observe()
            learn = np.random.choice([True, False], p=[(1 - alpha), alpha])
            if learn:
                self.embody_learner()
                query = self.teacher.plan(self.scenario.trajectories[trajectory_index])
                x, y, angle = query[0]
                loss = self.learner.learn(observation, np.array([2 * angle / pi]))
                self.learner.execute(angle)
                self.progress_bar.set_postfix({'loss': loss, 'alpha': alpha, 'learning': learn})
            else:  # exploit
                self.world.active_agent = self.learner
                prediction = self.learner.exploit(observation)
                self.learner.execute(prediction[0] * pi / 2)
                self.progress_bar.set_postfix({'alpha': alpha, 'learning': learn})

            training = not self.scenario.in_trajectory_goal(self.learner.x, self.learner.y) \
                       and self.scenario.in_scenario(self.learner.x, self.learner.y)

        return self.scenario.in_trajectory_goal(self.learner.x, self.learner.y)

    def embody_learner(self):
        # place teacher on learner's internal state
        self.teacher.x = self.learner.x
        self.teacher.y = self.learner.y
        self.teacher.v = self.learner.v
        self.teacher.theta = self.learner.theta

        self.world.active_agent = self.teacher

    def embody_teacher(self):
        # place learner on teacher's internal state
        self.learner.x = self.teacher.x
        self.learner.y = self.teacher.y
        self.learner.v = self.teacher.v
        self.learner.theta = self.teacher.theta

        self.world.active_agent = self.learner

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
                mean, aleatoric, epistemic, mixture, _ = self.learner.exploit(self.world.observe())
                self.learner.execute(mean * pi / 2)
                axis[0][0].scatter(i, aleatoric)
                axis[0][1].scatter(i, epistemic)
                axis[0][2].scatter(i, mixture)
                axis[1][0].scatter(i, mean * pi / 2)
                # # axis[1][1].scatter(i, acos(cos_0[2]))
                # axis[1][2].scatter(i, cos_theta[3])

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
