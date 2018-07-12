from .regime import ImitationLearningRegime


class OnlineImitationLearning(ImitationLearningRegime):

    def learn(self):
        prediction = self.learner.exploit(self.world.observe())
        query = self.teacher.exploit(self.world.state())
        self.learner.learn(self.world.observe(), query)
        return prediction, query

    def explore(self):
        self.learner.explore(self.world.state())  # TODO: T time-steps
        query = self.teacher.exploit(self.world.state())
        self.learner.learn(self.world.observe(), query)
        return query

