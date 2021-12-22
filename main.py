import numpy as np
from skimage.io import imread
import matlab
import os
from WatermarkEmbedding import WatermarkEmbedding
from ImageWork import LoadNamesImage, LoadImage, LoadWaterMark
from MatLabCalculation import LiftingWaveletTransform, Transform_Matlab_to_NP
from PIL import Image


def test(cv, res_cv):
    test = (cv == res_cv)
    return test


if __name__ == '__main__':
    water_mark = LoadWaterMark.load("Water Mark Image/wolfwater.jpg")  # считывание водяного знака

    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name('TestDataSet')

    load_image = LoadImage(path_name_image)

    mat_lab_lwt2 = LiftingWaveletTransform()
    scheme_embedding = WatermarkEmbedding(water_mark)
    transformator = Transform_Matlab_to_NP()

    i = 0
    number_image = 0

    try:
        while True:

            image = load_image.next_image()
            copy_image=image.copy()

            CA, CH, CV, CD = mat_lab_lwt2.lwt2(image)

            cv = transformator.get_NP(CV)

            cv_water = scheme_embedding.embed(cv)

            CV_water = transformator.get_MatLab_matrix(cv_water)

            sourse_image = mat_lab_lwt2.ilwt2(CA, CH, CV_water, CD, )

            image_np = transformator.get_NP(sourse_image)

            img = Image.fromarray(image_np.astype(np.uint8))

            test_image=copy_image-image_np

            # name_image = "CW" + str(number_image)
            # if (os.path.exists(rf"Image With WakterMark/{name_image}.tif")):
            #     os.remove(rf"Image With WaterMark/{name_image}.tif")
            # img.save(rf"Image With WaterMark/{name_image}.tif")
            # img.close()
            # number_image += 1

            print(f"картинок обработано {i}")
            i += 1

    except StopIteration:
        print("все изображения считаны")

    except Exception:
        print("ЧТО - ТО СЛУЧИЛОСЬ ")

    finally:
        print("выкл matlab")
        mat_lab_lwt2.exit_engine()

    print()
