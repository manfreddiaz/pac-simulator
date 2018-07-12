
class World:
    def __init__(self, width, height, active_agent=None):
        self.width = width
        self.height = height
        self.semantics = []
        self.active_agent = active_agent

    def state(self):
        pass

    def observe(self):
        pass


class VisualWorld(World):
    def __init__(self, width, height):
        World.__init__(self, width, height)
        self.visuals = []
        self.engine = None

    def set_engine(self, engine):
        self.engine = engine
        self.engine.set_world(self)

    def render(self):
        for visual in self.visuals:
            visual.render()
        self.engine.render()

    def update(self):
        for visual in self.visuals:
            visual.update()
        self.engine.update()

    def observe(self):
        self.update()
        return self.engine.visualization()

    def state(self):
        return self.semantics, self.engine.visualization()
