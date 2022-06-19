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
from AttackInImage import *
from CreateFeatureVector import *

def f1(pr , re):
    f1= 2* ((pr*re)/(pr+re))
    return f1

def acc(zz ,ff , total=1024):
    return  (zz+ff)/total


def calcW(w1 = 1, w2 = 1 , w3 = 1, w4 = 1 ):
    t0=4*w1
    t1=w1*w2
    t2=w2*w3
    t3=w3*w4
    tf=w3*2
    res = t0+t1+t2+t3+tf
    print(res)
    return res

if __name__ == '__main__':
    calcW(3,3,3)





