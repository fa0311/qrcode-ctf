import numpy as np


class ColorBase:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b

    @property
    def ndarray(self):
        return np.array([self.b, self.g, self.r], dtype=np.uint8)


class Color(ColorBase):
    white = ColorBase(255, 255, 255)
    black = ColorBase(0, 0, 0)
    red = ColorBase(255, 0, 0)
    green = ColorBase(0, 255, 0)
    blue = ColorBase(0, 0, 255)
