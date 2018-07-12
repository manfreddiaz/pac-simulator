from simulator import Agent


class LearningAgent(Agent):
    def explore(self, state, horizon: int = 1):
        pass

    def learn(self, state, action):
        pass

    def exploit(self, state, horizon: int = 1):
        pass

    def create_model(self):
        pass

