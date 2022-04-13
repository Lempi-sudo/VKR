import matlab.engine
import numpy as np
from ImageWork import ImagesNamesLoader, ImageLoader
from skimage.io import imread
import matlab
import pandas as pd
import numpy as np
import cv2
from ImageWork import WaterMarkLoader
from Metrici import psnr
import io
import collections

#результат выдает побитовое сравнение в процентах насколько похожи два ЦВЗ
def pobitovo_sravnenie_WaterMark(W1, W2 , total_bit=1024):
    t1 = W1==W2
    return t1.sum()/total_bit*100



if __name__ == '__main__':
    path="Water Mark Image/waterMark1.jpg"
    W1=WaterMarkLoader.load(path)
    W2=WaterMarkLoader.load(path)
    W1[0:70]=0




    print(pobitovo_sravnenie_WaterMark(W1,W2))



