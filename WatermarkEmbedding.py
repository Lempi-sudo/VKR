import numpy as np

    class GenerateBinaryWaterMark:
        def water_mark(self, size=32*32):
            return np.random.randint(2, size=size)


    class WatermarkEmbedding():

        def __init__(self ,f, W ):
            self.W=W
            self.f=f


        def embed(self):
            pass








