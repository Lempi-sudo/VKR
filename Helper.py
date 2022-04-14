import numpy as np
from PIL import Image


def GenerateWaterMark(path_save = rf"Water Mark Image/WaterMarkRandom.tif"):
    wm = np.random.randint(0, 2, (32, 32))
    wm[wm>0]=255
    img = Image.fromarray(wm.astype(np.uint8))
    img.save(path_save)
    img.close()
