import io
import threading

import numpy as np
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# from PIL import Image
from .renderer import Renderer


class PyplotRenderer(Renderer):
    def show(self):
        pass
        # plt.ion()
        # plt.show()

    def __init__(self, width, height):
        Renderer.__init__(self, width, height)
        self.figure = Figure(figsize=(width, height), dpi=300)
        self.canvas = FigureCanvas(figure=self.figure)
        self.axis = self.figure.gca()
        self.buffer = io.BytesIO()
        self.lock = threading.Lock()
        # self.writer = animation.writers['ffmpeg'](24)
        self.create()

    def create(self):
        self.set_background((0.0, 0.0, 0.0))
        self.set_aspect_ratio()
        self.set_canvas_size(self.width, self.height)
        self.set_axis_shown()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axis.get_xaxis().set_major_locator(NullLocator())
        self.axis.get_yaxis().set_major_locator(NullLocator())
        self.axis.margins(0, 0)
        self.figure.tight_layout(pad=0)
        # self.writer.setup(self.figure, 'training.mp4', self.figure.get_dpi())

    def set_background(self, color):
        self.figure.patch.set_facecolor(color)
        self.axis.patch.set_facecolor(color)

    def set_aspect_ratio(self, ratio=1):
        self.axis.set_aspect(ratio)

    def set_canvas_size(self, width, height):
        self.axis.set(xlim=[-width, width], ylim=[-height, height])

    def set_axis_shown(self, value=False):
        self.axis.get_xaxis().set_visible(value)
        self.axis.get_yaxis().set_visible(value)

    def render(self):
        for simulation_object in self.simulation_objects:
            self.axis.add_artist(simulation_object)
        self.update()

    def update(self):
        with self.lock:
            self.canvas.draw()
            # plt.pause(0.01)
        # self.writer.grab_frame()

    def context(self):
        with self.lock:
            image = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8')
            width, height = self.figure.get_size_inches() * self.figure.get_dpi()
            image = image.reshape((int(height), int(width), 3))
            # plt.imsave('output.png', image)
            return image
            # self.buffer.seek(0)
            # self.figure.savefig(self.buffer, format='png') # bbox_inches='tight', pad_inches=0, facecolor=(0.0, 0.0, 0.0))
            # image = Image.open(self.buffer)
            # image = image.convert('RGB')
            # image = np.array(image)
            # return image
