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


if __name__ == '__main__':
    intruder = Attack()
    intruder.frame_replacement("CW","AttackedImage/Replace_frame" , size=100 , path_image_for_replace="nno")




