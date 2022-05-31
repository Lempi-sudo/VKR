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


if __name__ == '__main__':
    print(f1(0.61,0.64))
    print(acc(72908,25161))





