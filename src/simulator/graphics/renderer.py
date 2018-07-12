
class Renderer:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.simulation_objects = []

    def render(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def context(self):
        raise NotImplementedError()

    def show(self):
        raise NotImplementedError()