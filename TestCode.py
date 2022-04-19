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



if __name__ == '__main__':
    path="Water Mark Image/waterMark1.jpg"
    PathC="Water Mark Image/waterMark3.jpg"
    PathCW="Water Mark Image/wrDota2.jpg"

    water_mark2=Helper.GenerateWaterMark(IsSave=True)
    print("sum2 = ", water_mark2.sum())

    C=WaterMarkLoader.load(PathC)
    CW=WaterMarkLoader.load(PathCW)
    C1 = imread(PathC)
    total_len=32*32
    B= pobitovo_sravnenie_WaterMark(C,CW ,total_len )
    PSNR = cv2.PSNR(C,CW)
    print()



