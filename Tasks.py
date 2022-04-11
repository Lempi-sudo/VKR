import numpy as np
from skimage.io import imread
import matlab
import os
from WatermarkEmbedding import WatermarkEmbedding
from ImageWork import ImagesNamesLoader, ImageLoader, WaterMarkLoader
from MatLabCalculation import LiftingWaveletTransform, Transform_Matlab_to_NP
from PIL import Image
from CreateFeatureVector import ImageFeature
from AttackInImage import Attack
from skimage.util import random_noise
from Metrici import psnr


def get_water_mark(path):
    water_mark = WaterMarkLoader.load(path)
    return water_mark

def LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_dir):
    ## Данная функция берет все изображения из директории path_dataSet
    ## и встраивает туда водяной знак в область виевлет преобразования 3 уровня
    ## и сохраняет в path_save_dir
    water_mark = WaterMarkLoader.load(path_waterMark)  # считывание водяного знака

    load_name = ImagesNamesLoader()
    path_name_image = load_name.get_image_name_list(path_dataSet)

    load_image = ImageLoader(path_name_image)

    mat_lab_lwt2 = LiftingWaveletTransform()
    scheme_embedding = WatermarkEmbedding(water_mark)
    transformator = Transform_Matlab_to_NP()

    i = 1
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

def all_attack():
    attack=Attack()
    attack.All_Attack()

#функция создает txt документ с таблицей  векторов признаков (путь - path_save_feature_vec_arg)
#вектора признаков формируются из картинок со встроенным ЦВЗ, после либо до атаки.
def create_feature(path_save_feature_vec_arg, path_image_water_arg, water_mark_arg):
    water_mark = water_mark_arg
    extract_feature = ImageFeature(water_mark)
    path_save_feature_vec = path_save_feature_vec_arg
    path_image_water = path_image_water_arg
    extract_feature.save_feature_data(path_save_feature_vec, path_image_water)
    extract_feature.close()

def all_feature():
    pathwaterMark = "Water Mark Image/dota2.jpg"
    water_mark = WaterMarkLoader.load(pathwaterMark)
    path_save_water_mark_image = 'CW'

    create_feature("feature_vec/AverageAttack.txt", "AttackedImage/AverageAttack", water_mark)
    create_feature("feature_vec/HistogramAttack.txt", "AttackedImage/HistogramAttack", water_mark)
    create_feature("feature_vec/GammaCorrection.txt", "AttackedImage/GammaCorrection", water_mark)
    create_feature("feature_vec/JPEG50.txt", "AttackedImage/JPEG50", water_mark)
    create_feature("feature_vec/medianAttack.txt", "AttackedImage/medianAttack", water_mark)
    create_feature("feature_vec/SaltPaperAttack.txt", "AttackedImage/SaltPaperAttack", water_mark)
    create_feature("feature_vec/Sharpness.txt", "AttackedImage/Sharpness", water_mark)


#1)	Обучить модель на 200 картинок с одним ЦВЗ и попробовать подсунуть обученной модели
#картинку с другим ЦВЗ и посмотреть на результат.
def Task1(path_waterMark, path_dataSet, path_save_dir , path_feature_vector ):
    LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_dir)
    create_feature(path_feature_vector, path_save_dir, get_water_mark(path_waterMark))


    path_dataSet2 = 'DataSet/DataSetTask1'  # путь до набора картинок
    path_save_CW2 = 'Task1CW/AnotherCW'
    path_feature_vec2 = "feature_vec/Task1/anotherCWFearVec.txt"
    water_mark_2=  "Water Mark Image/waterMark1.jpg"

    #LWT2EmbedWaterMark(water_mark_2, path_dataSet2, path_save_CW2)
    #create_feature(path_feature_vec2, path_save_CW2, get_water_mark(water_mark_2))

#Task 2: Обучить на встроенной цвз с малым содержанием 0 - ых или малым количеством 1 -ых битов.
def Task2():
    path_waterMark = "Water Mark Image/WhiteWaterMark.jpg"
    path_dataSet = 'DataSet' # путь до набора картинок
    path_save_CW = 'Task2CW'
    path_feature_vec= "feature_vec/Task2/Task2FeatureVec.txt"

    #LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_CW)
    #create_feature(path_feature_vec, path_save_CW, get_water_mark(path_waterMark))

    path_dataSet2 = 'DataSet/Task2'  # путь до набора картинок
    path_save_CW2 = 'Task2CW/AnotherCW'
    path_feature_vec2 = "feature_vec/Task2/anotherCWFearVec.txt"
    water_mark_2 = "Water Mark Image/waterMark1.jpg"

    LWT2EmbedWaterMark(water_mark_2, path_dataSet2, path_save_CW2)
    create_feature(path_feature_vec2, path_save_CW2, get_water_mark(water_mark_2))
