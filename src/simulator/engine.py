from .graphics import Renderer, PyplotRenderer


class Engine:
    def __init__(self):
        self.renderer = None
        self.world = None

    def set_world(self, world):
        if self.world is None:
            self.world = world
        if self.renderer is None:
            self.renderer = PyplotRenderer(self.world.width, self.world.height)

    def render(self):
        for visual in self.world.visuals:
            self.renderer.simulation_objects.append(visual.graphics)
        self.renderer.render()
        self.renderer.show()

    def update(self):
        self.renderer.update()

    def visualization(self):
        return self.renderer.context()
