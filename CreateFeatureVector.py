from scipy.ndimage import median_filter
from ImageWork import LoadNamesImage, LoadImage, LoadWaterMark
import os
from PIL import Image
import numpy as np
import pandas as pd
from MatLabCalculation import LiftingWaveletTransform, Transform_Matlab_to_NP

# df = pd.DataFrame.from_dict({'bit': [], 'feature': []})
# print(df)
#
# arr = np.array([1, 2, 3, 4])
#
# for i in range(10):
#     new_row = {"bit": int(i), "feature": arr}
#     df = df.append(new_row, ignore_index=True)
#
# print(df)
# df.to_csv('feature_vec/Image_with_water.txt')


class ImageFeature:

    def __init__(self, water_mark):
        self.water_mark = water_mark
        self.mat_lab = LiftingWaveletTransform()
        self.load_name = LoadNamesImage()
        self.transformator = Transform_Matlab_to_NP()

    def __extract_hl2_area__(self, f, intex=(128, 192, 64, 128)):
        hl2 = f[intex[0]:intex[1], intex[2]:intex[3]]
        return hl2

    def __all_blocks__(self, hl2):
        size = hl2.shape[0]
        blocks = []
        for i in range(0, size, 2):
            for j in range(0, size, 2):
                block = hl2[i:i + 2, j:j + 2]
                blocks.append(block)
        return blocks

    def save_feature_data(self, path_save, path_image):
        path_name_image = self.load_name.get_list_image_name(path_image)
        load_image = LoadImage(path_name_image)

        i = 1


        df = pd.DataFrame.from_dict({'bit': [], 'feature_vec': []})
        try:
            while True:
                image = load_image.next_image()

                CA, CH, CV, CD = self.mat_lab.lwt2(image)

                cv = self.transformator.get_NP(CV)

                hl2 = self.__extract_hl2_area__(cv)

                blocks = self.__all_blocks__(hl2)

                iter_water = iter(self.water_mark)
                bit_i=1
                for block in blocks:
                    vec = np.ravel(block)
                    print(f"bit_i {bit_i} ")
                    bit_i+=1
                    new_row = {'bit': next(iter_water), "feature_vec": vec}
                    df=df.append(new_row, ignore_index=True)

                print(f"векторов признаков сформировано {i}")
                i += 1



        except StopIteration:
            print("все изображения считаны")

        df.to_csv(path_save)

    def close(self):
        self.mat_lab.exit_engine()
