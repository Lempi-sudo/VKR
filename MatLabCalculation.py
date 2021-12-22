import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matlab.engine
import numpy as np
from ImageWork import LoadNamesImage, LoadImage
from skimage.io import imread
import matlab


class LiftingWaveletTransform:

    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.CA = 0
        self.CD = 0
        self.CH = 0
        self.CV = 0

    def lwt2(self,image):
        try:
            mat_image = matlab.double(image.tolist())
            [CA, CH, CV, CD] = self.eng.lwt2(mat_image, 'haar', 3, nargout=4)
            self.CA = CA
            self.CD = CD
            self.CH = CH
            self.CV = CV
        except Exception:
            print("ошибка в def lwt2(self,image): ")

        return self.CA , self.CH ,self.CV ,self.CD


    def ilwt2(self,CA, CH, CV, CD,):
        return self.eng.ilwt2(CA, CH, CV, CD, 'haar', 3, nargout=1)

    def exit_engine(self):
        self.eng.exit()



class Transform_Matlab_to_NP():
    def get_NP(self, matx):
        return np.array(matx)

    def get_MatLab_matrix(self, np_arr):
        return matlab.double(np_arr.tolist())
