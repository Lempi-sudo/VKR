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


if __name__ == '__main__':
    pathwaterMark = "Water Mark Image/WaterMarkRandom.jpg"
    water_mark = WaterMarkLoader.load(pathwaterMark)
    f= ImageFeature(water_mark)
    f.save_all_image_feature_data("feature_vec/Test.txt","Task7/Img")




