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


def get_water_mark(path):
    water_mark = WaterMarkLoader.load(path)
    return water_mark

def LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_dir , Treshold = 3.46):
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
            c = transformator.get_NP(CA)

            cobj = scheme_embedding.embed(c)

            Cobj_water = transformator.get_MatLab_matrix(cobj)

            sourse_image = mat_lab_lwt2.ilwt2(Cobj_water, CH, CV, CD )

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

# def all_feature():
#     pathwaterMark = "Water Mark Image/waterMark3.jpg"
#     water_mark = WaterMarkLoader.load(pathwaterMark)
#
#     create_feature("feature_vec/AverageAttackDota.txt", "ImgWMReableAttack/AverageAttack", water_mark)
#     create_feature("feature_vec/HistogramAttackDota.txt", "ImgWMReableAttack/HistogramAttack", water_mark)
#     create_feature("feature_vec/GammaCorrectionDota.txt", "ImgWMReableAttack/GammaCorrection", water_mark)
#     create_feature("feature_vec/JPEG50Dota.txt", "ImgWMReableAttack/JPEG50", water_mark)
#     create_feature("feature_vec/medianAttackDota.txt", "ImgWMReableAttack/medianAttack", water_mark)
#     create_feature("feature_vec/SaltPaperAttackDota.txt", "ImgWMReableAttack/SaltPaperAttack", water_mark)
#     create_feature("feature_vec/SharpnessDota.txt", "ImgWMReableAttack/Sharpness", water_mark)
#     create_feature("feature_vec/NO_AttackDota.txt", "ImgWMReadble", water_mark)
###################################
# create_feature("feature_vec/AverageAttack.txt", "AttackedImage/AverageAttack", water_mark)
# create_feature("feature_vec/HistogramAttack.txt", "AttackedImage/HistogramAttack", water_mark)
# create_feature("feature_vec/GammaCorrection.txt", "AttackedImage/GammaCorrection", water_mark)
# create_feature("feature_vec/JPEG50.txt", "AttackedImage/JPEG50", water_mark)
# create_feature("feature_vec/medianAttack.txt", "AttackedImage/medianAttack", water_mark)
# create_feature("feature_vec/SaltPaperAttack.txt", "AttackedImage/SaltPaperAttack", water_mark)
# create_feature("feature_vec/Sharpness.txt", "AttackedImage/Sharpness", water_mark)
# create_feature("feature_vec/NO_Attack.txt", "CW", water_mark)

def all_feature():
    pathwaterMark = "Water Mark Image/WaterMarkRandom.jpg"
    water_mark = WaterMarkLoader.load(pathwaterMark)

    create_feature("feature_vec/AverageAttack.txt", "AttackedImage/AverageAttack", water_mark)
    create_feature("feature_vec/HistogramAttack.txt", "AttackedImage/HistogramAttack", water_mark)
    create_feature("feature_vec/GammaCorrection.txt", "AttackedImage/GammaCorrection", water_mark)
    create_feature("feature_vec/JPEG50.txt", "AttackedImage/JPEG50", water_mark)
    create_feature("feature_vec/medianAttack.txt", "AttackedImage/medianAttack", water_mark)
    create_feature("feature_vec/SaltPaperAttack.txt", "AttackedImage/SaltPaperAttack", water_mark)
    create_feature("feature_vec/Sharpness.txt", "AttackedImage/Sharpness", water_mark)
    create_feature("feature_vec/NO_Attack.txt", "CW", water_mark)

#1)	Обучить модель на 200 картинок с одним ЦВЗ и попробовать подсунуть обученной модели
#картинку с другим ЦВЗ и посмотреть на результат.
def Task1(path_waterMark, path_dataSet, path_save_dir, path_feature_vector):
    #LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_dir)
    #create_feature(path_feature_vector, path_save_dir, get_water_mark(path_waterMark))


    water_mark_2 = "Water Mark Image/WaterMarkRandom.jpg" # путь до водяного знака3 (Дота2)
    path_dataSet_2 = 'DataSet/DataSetTask1' # путь до набора картинок
    path_save_CW_2 = 'dota2'
    path_feature_vec_2= "feature_vec/dota2WMFeatVec.txt"

    LWT2EmbedWaterMark(water_mark_2, path_dataSet_2, path_save_CW_2)
    create_feature(path_feature_vec_2, path_save_CW_2, get_water_mark(water_mark_2))

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
#Обучать нейросеть на картинках без искажений смотреть на метрики,
# потом дообучить с искажениями и смотреть на метрики, хорошо бы чтобы они улучшились
def Task4():
    path_waterMark = "Water Mark Image/WaterMarkRandom.jpg"
    path_dataSet = 'DataSet'  # путь до набора картинок
    path_save_CW = 'CW'
    path_feature_vec = "feature_vec/Task2/Task2FeatureVec.txt"

    #ВСТРАИВАНИЕ ЦВЗ В ИЗОБРАЖЕНИЯ
    #LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_CW)

    #АТАКИ НА ИЗОБРАЖЕНИЯ С ЦВЗ
    all_attack()

    #СОЗДАНИЕ ВЕКТОРОВ ПРИЗНАКОВ
    #all_feature()

    #Далее необходимо обучить сеть на изображениях без атак а затем дообучить на изображениях с атаками

    #СРАВНЕНИЯ ИЗОБРАЖЕНИЙ
    W=WaterMarkLoader.load("Water Mark Image/WaterMarkRandom.jpg")
    WR=WaterMarkLoader.load("W_R/WR_SP.tif")

    print(pobitovo_sravnenie_WaterMark(W,WR))

def embed_wm_differn_T(threshold_list_arg):
    threshold_list = threshold_list_arg
    path_water_mark = "Water Mark Image/WaterMarkRandom.jpg"
    dirpath = "Task7/PSNR at T/T = "
    # создаём нужные директории
    for t in threshold_list:
        path_save = dirpath + str(t)
        if not (os.path.exists(path_save)):
            os.mkdir(path_save)

    #вставялем цвз
    for T in threshold_list:
        dirsave = dirpath + str(T)
        print("Порог = ", T)
        print("сохранение в папку ", dirsave)
        LWT2EmbedWaterMark(path_water_mark, "Task7/Img", dirsave, T)

def dependens_PSNR_and_T(threshold_list):

    threshold_list = threshold_list

    name_loader = ImagesNamesLoader()
    path_image_name_no_wm = name_loader.get_image_name_list("Task7/Img")
    image_no_wm=[]

    for path in path_image_name_no_wm:
        image_no_wm.append(imread(path))

    count_image=len(image_no_wm)

    dirpath = "Task7/PSNR at T/T = "
    listrow = []

    for T in threshold_list:
        dir = dirpath + str(T)
        print(rf"{dir}     ПОРОГ {T}")
        inl = name_loader.get_image_name_list(dir)
        image_wm=[]
        for path in inl:
            image_wm.append(imread(path))

        list_value_psnr = []

        for i , path in enumerate(inl , start=0):
            PSNR = cv2.PSNR(image_wm[i], image_no_wm[i])
            indexstart = path.find("CW")
            print(rf" psnr = {PSNR} . Картинка № {path[indexstart:]}")
            list_value_psnr.append(PSNR)

        new_row = {'Threshold': T, "img1": round(list_value_psnr[0],1), "img2": round(list_value_psnr[1],1), "img3": round(list_value_psnr[2],1),
               "img4": round(list_value_psnr[3],1)}

        listrow.append(new_row)

    columns = ['Threshold', 'img1', 'img2', 'img3', 'img4'  ]
    df = pd.DataFrame(data=listrow, columns=columns)
    df.to_excel("Task7/Psnr at T.xlsx")
    df.to_csv("Task7/Psnr at t.txt")

#Зависимость метрик (PSNR и ещё че ни будь) от порога (T) для разныхх картинов типа лена оронгутанг танк перцы и тд .
def Task7():
    threshold_list = np.arange(0, 170, 2)
    embed_wm_differn_T(threshold_list)
    dependens_PSNR_and_T(threshold_list)


#Task9: проверить метрики при вставке цвз в разнвые части (hh ll lh hl)
def Task9():
    path_waterMark = "Water Mark Image/WaterMarkRandom.jpg"
    path_dataSet = 'DataSet'  # путь до набора картинок
    path_save_CW = 'CW'

    #ВСТРАИВАНИЕ ЦВЗ В ИЗОБРАЖЕНИЯ
    LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_CW,Treshold=12)

    #АТАКИ НА ИЗОБРАЖЕНИЯ С ЦВЗ
    all_attack()

    #СОЗДАНИЕ ВЕКТОРОВ ПРИЗНАКОВ
    all_feature()



def LWT2forTask10(path_dataSet, path_save_dir , number_image , level):
    load_name = ImagesNamesLoader()
    path_name_image = load_name.get_image_name_list(path_dataSet)

    load_image = ImageLoader(path_name_image)

    mat_lab_lwt2 = LiftingWaveletTransform()
    transformator = Transform_Matlab_to_NP()

    i = 1
    number_image = number_image
    try:
        while True:
            image = load_image.next_image()

            CA, CH, CV, CD = mat_lab_lwt2.lwt2(image , level=level)
            # [ll,lh,hl,hh] = lwt2(x)

            image_np = transformator.get_NP(CA)

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

#как выглядит виевлет трансформ для разных уровней декомпозиции
def Task10():
    path_save_CW = "sampleBlock"
    path_dataSet= "Task7/Img"
    for i in range(1,7,1):
        LWT2forTask10(path_dataSet, path_save_CW,i,i)



