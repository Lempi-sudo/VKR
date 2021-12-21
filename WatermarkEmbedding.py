import numpy as np


class GenerateBinaryWaterMark:
    @staticmethod
    def water_mark(size=32 * 32):
        return np.random.randint(2, size=size)


class WatermarkEmbedding:
    def __init__(self, f, w):
        self.W = w
        self.f = f

    def embed(self):
        pass
