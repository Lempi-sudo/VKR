from Tasks import Task1, Task4 , Task7, Task9,  LWT2EmbedWaterMark , all_attack, all_feature
from AttackInImage import *
import numpy as np
from WatermarkEmbedding import WatermarkEmbedding
from ImageWork import *
from MatLabCalculation import LiftingWaveletTransform, Transform_Matlab_to_NP
from PIL import Image
from CreateFeatureVector import ImageFeature
from AttackInImage import Attack
from skimage.util import random_noise
from Metrici import psnr , pobitovo_sravnenie_WaterMark
import cv2
from ImageWork import *
import pandas as pd



def embed_readble_wm():
    path_waterMark = "Water Mark Image/waterMark3.jpg"
    path_dataSet = 'DataSetTest'
    path_save_water_mark_image = 'ImgWMReadble'
    LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_water_mark_image)

def LWT2EmbedWaterMarkcopy(path_waterMark, path_dataSet, path_save_dir , Treshold = 3.46):
    ## Данная функция берет все изображения из директории path_dataSet
    ## и встраивает туда водяной знак в область виевлет преобразования 3 уровня
    ## и сохраняет в path_save_dir
    water_mark = WaterMarkLoader.load(path_waterMark)  # считывание водяного знака

    load_name = ImagesNamesLoader()
    path_name_image = load_name.get_image_name_list(path_dataSet)

    load_image = ImageLoader(path_name_image)

    mat_lab_lwt2 = LiftingWaveletTransform()
    scheme_embedding = WatermarkEmbedding(water_mark, Treshold)
    transformator = Transform_Matlab_to_NP()

    i = 1
    number_image = 1
    try:
        while True:
            image = load_image.next_image()

            CA, CH, CV, CD = mat_lab_lwt2.lwt2(image)
            # [ll,lh,hl,hh] = lwt2(x)
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

def All_Attack_ReadbleImg():
    attack = Attack()

    image_path='ImgWMReadble'
    image_attacked_salt_paper='ImgWMReableAttack/SaltPaperAttack'
    image_attacked_median='ImgWMReableAttack/medianAttack'
    image_attacked_average = 'ImgWMReableAttack/AverageAttack'
    image_save_jpeg20 = 'ImgWMReableAttack/JPEG20'
    image_save_jpeg30 = 'ImgWMReableAttack/JPEG30'
    image_save_jpeg40 = 'ImgWMReableAttack/JPEG40'
    image_save_jpeg50 = 'ImgWMReableAttack/JPEG50'
    image_histogram = 'ImgWMReableAttack/HistogramAttack'
    Gamma_Correction = 'ImgWMReableAttack/GammaCorrection'
    Sharpness = 'ImgWMReableAttack/Sharpness'

    attack.median_attack(path_image=image_path , path_image_attacked=image_attacked_median )
    attack.salt_peper_attack(path_image=image_path , path_image_attacked= image_attacked_salt_paper)
    attack.average_filter(path_image=image_path, path_image_attacked=image_attacked_average)
    attack.Save_JPEG(image_path, image_save_jpeg50, 50)
    attack.Histogram(image_path,image_histogram)
    attack.Gamma_Correction(image_path, Gamma_Correction)
    attack.Sharpness(image_path, Sharpness)



if __name__ == '__main__':
    Task7()

    # path_waterMark = "Water Mark Image/WaterMarkRandom.jpg" # путь до водяного знака3 (Дота2)
    # path_dataSet = 'DataSet/AnotherCWTask1/' # путь до набора картинок
    # path_save_dir = 'CW'
    # path_feature_vector= "feature_vec/RandomWMFeatVec.txt"


    #embed_readble_wm()
    #All_Attack_ReadbleImg()



    #all_feature()


    # path_waterMark = "Water Mark Image/waterMark1.jpg"
    # path_dataSet = 'DataSet'
    # path_save_water_mark_image = 'CW'
    #
    # path_W_tilda="W_R/histogram.tif"
    # w_tilda=imread(path_W_tilda)
    #
    # w_tilda[w_tilda < 100] = 0
    # w_tilda[w_tilda >= 100] = 1
    # w=WaterMarkLoader.load(path_waterMark)
    # print(psnr(w_tilda,w))


    #LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_water_mark_image)

    #all_attack()

    # pathwaterMark = "Water Mark Image/dota2.jpg"
    # water_mark = LoadWaterMark.load(pathwaterMark)
    # create_feature("feature_vec/test.txt", "Test", water_mark)


