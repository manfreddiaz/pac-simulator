import matplotlib.lines as lines

from ....simulator.graphics import Visual


class LinePyplot(Visual):
    def __init__(self, line, width=1):
        self.line = line
        self.width = width
        self.graphics = lines.Line2D(xdata=[line.start[0], line.end[0]], ydata=[line.start[1], line.end[1]],
                                     linewidth=width, color=line.color, zorder=0)

        if line.discontinuous:
            self.graphics.set_linestyle('--')

    def update(self):
        self.graphics.set_xdata([self.line.start[0], self.line.end[0]])
        self.graphics.set_ydata([self.line.start[1], self.line.end[1]])
