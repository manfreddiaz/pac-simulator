
class Task:

    def __init__(self, name, world, oracle):
        self.name = name
        self.world = world
        self.oracle = oracle
        self.learners = []  # list<Agent>

    def demo(self):
        return self.oracle.exploit(self.world.state())
