class Agent:
    def __init__(self, world, x=0, y=0, theta=0.0, v=0.0):
        self.x = x
        self.y = y
        self.v = v
        self.theta = theta
        self.world = world
        self.ghost = (x, y, theta, v)

    def reset(self):
        self.x = self.ghost[0]
        self.y = self.ghost[1]
        self.theta = self.ghost[2]
        self.v = self.ghost[3]

    def learn(self, state, action):
        raise NotImplementedError()

    def exploit(self, state, horizon=1):
        raise NotImplementedError()

    def explore(self, state, horizon=1):
        raise NotImplementedError()

    def execute(self, action):
        raise NotImplementedError()
