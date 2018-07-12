
class Line:
    def __init__(self, start, end, color=(1.0, 1.0, 1.0), discontinuous=False):
        self.start = start
        self.end = end
        self.color = color
        self.discontinuous = discontinuous