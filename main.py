import numpy as np
from skimage.io import imread
import matlab
import os
from WatermarkEmbedding import WatermarkEmbedding
from ImageWork import LoadNamesImage, LoadImage, LoadWaterMark
from MatLabCalculation import LiftingWaveletTransform, Transform_Matlab_to_NP
from PIL import Image
from CreateFeatureVector import ImageFeature
from AttackInImage import Attack
from skimage.util import random_noise
from Metrici import psnr


def LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_dir):
    ## Данная функция берет все изображения из директории path_dataSet
    ## и встраивает туда водяной знак в область виевлет преобразования 3 уровня
    ## и сохраняет в path_save_dir
    water_mark = LoadWaterMark.load(path_waterMark)  # считывание водяного знака

    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name(path_dataSet)

    load_image = LoadImage(path_name_image)

    mat_lab_lwt2 = LiftingWaveletTransform()
    scheme_embedding = WatermarkEmbedding(water_mark)
    transformator = Transform_Matlab_to_NP()

    i = 0
    number_image = 1
    try:
        while True:
            image = load_image.next_image()

            CA, CH, CV, CD = mat_lab_lwt2.lwt2(image)

            cv = transformator.get_NP(CV)

            cv_water = scheme_embedding.embed(cv)

            CV_water = transformator.get_MatLab_matrix(cv_water)

            sourse_image = mat_lab_lwt2.ilwt2(CA, CH, CV_water, CD, )

            image_np = transformator.get_NP(sourse_image)

            img = Image.fromarray(image_np.astype(np.uint8))

            name_image = "CW" + str(number_image)

            path_save = rf"{path_save_dir}/{name_image}.tif"

            if (os.path.exists(path_save)):
                os.remove(path_save)
            img.save(path_save)
            img.close()
            number_image += 1

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


def create_feature(path_save_feature_vec_arg, path_image_water_arg, water_mark_arg):
    water_mark = water_mark_arg
    extract_feature = ImageFeature(water_mark)
    path_save_feature_vec = path_save_feature_vec_arg
    path_image_water = path_image_water_arg
    extract_feature.save_feature_data(path_save_feature_vec, path_image_water)
    extract_feature.close()

def all_feature():
    pathwaterMark = "Water Mark Image/dota2.jpg"
    water_mark = LoadWaterMark.load(pathwaterMark)
    path_save_water_mark_image = 'CW'

    create_feature("feature_vec/AverageAttack.txt", "AttackedImage/AverageAttack", water_mark)
    create_feature("feature_vec/HistogramAttack.txt","AttackedImage/HistogramAttack" , water_mark)
    create_feature("feature_vec/GammaCorrection.txt", "AttackedImage/GammaCorrection", water_mark)
    create_feature("feature_vec/JPEG50.txt","AttackedImage/JPEG50" , water_mark)
    create_feature("feature_vec/medianAttack.txt","AttackedImage/medianAttack" , water_mark)
    create_feature("feature_vec/SaltPaperAttack.txt","AttackedImage/SaltPaperAttack" , water_mark)
    create_feature("feature_vec/Sharpness.txt","AttackedImage/Sharpness" , water_mark)

def all_attack():
    attack=Attack()
    attack.All_Attack()



if __name__ == '__main__':
    path_waterMark = "Water Mark Image/dota2.jpg"
    path_dataSet = 'DataSet'
    path_save_water_mark_image = 'CW'

    path_W_tilda="W_Tilda/histogram.tif"

    w_tilda=imread(path_W_tilda)
    w_tilda[w_tilda < 100] = 0
    w_tilda[w_tilda >= 100] = 1
    w=LoadWaterMark.load(path_waterMark)
    print(psnr(w_tilda,w))


    #LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_water_mark_image)

    #all_attack()

    # pathwaterMark = "Water Mark Image/dota2.jpg"
    # water_mark = LoadWaterMark.load(pathwaterMark)
    # create_feature("feature_vec/test.txt", "Test", water_mark)
    #

