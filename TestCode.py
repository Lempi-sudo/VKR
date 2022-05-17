import matlab.engine
import numpy as np
from ImageWork import ImagesNamesLoader, ImageLoader
from skimage.io import imread
import matlab
import pandas as pd
import numpy as np
import cv2
from ImageWork import WaterMarkLoader
from Metrici import psnr , pobitovo_sravnenie_WaterMark
from skimage.io import imread
import io
import collections
import WatermarkEmbedding as we
import Helper
from WatermarkEmbedding import *



if __name__ == '__main__':
    arr = np.zeros((256,256))
    water_mark = WaterMarkLoader.load("Water Mark Image/WaterMarkRandom.jpg")  # считывание водяного знака
    wm = WatermarkEmbedding(water_mark)
    b = wm.embed_in_all_image(arr)



