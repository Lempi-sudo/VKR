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



#Неточность может возникать в ilwt2 или в save()
def test_lwt2_before_after_save(): #плохие тесты почему то разные значения для казалось бы одинаковых преобразованиях
    water_mark = LoadWaterMark.load(path_waterMark)  # считывание водяного знака
    image = imread("TestDataSet/Image00001.tif")
    mat_lab_lwt2 = LiftingWaveletTransform()
    transformator = Transform_Matlab_to_NP()
    cheme_embedding = WatermarkEmbedding(water_mark)
    CA, CH, CV, CD = mat_lab_lwt2.lwt2(image)
    cv = transformator.get_NP(CV)
    cv_copy=cv.copy()

    cv_water = cheme_embedding.embed(cv)

    cv_water_copy=cv_water.copy()

    CV_water = transformator.get_MatLab_matrix(cv_water)

    cwtilda = mat_lab_lwt2.ilwt2(CA, CH, CV_water, CD, )

    cw_tilda_np=transformator.get_NP(cwtilda)

    test_image=cw_tilda_np-image

    CA2, CH2, CV2, CD2 = mat_lab_lwt2.lwt2(cwtilda)

    cv_tilda = transformator.get_NP(CV2)

    test=np.abs(cv_water_copy-cv_tilda)

    print(test)

def Test_LWT2Embed():

    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name('TestImagewithWaterMark')
    load_image = LoadImage(path_name_image)
    images_water_mark=[]
    sourse_images_water_mark=[]

    try:
        while True:

            image = load_image.next_image()

            image2=image.copy()

            sourse_images_water_mark.append(image2)
            images_water_mark.append(image)

    except StopIteration:
        print("все изображения считаны")

    except Exception:
        print("ЧТО - ТО СЛУЧИЛОСЬ ")

    finally:
        print("Закончил создание массива картинок ")



    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name('TestImage')
    load_image = LoadImage(path_name_image)
    images = []
    sourse_images = []

    try:
        while True:
            image = load_image.next_image()
            image3=image.copy()

            sourse_images.append(image3)
            images.append(image)

    except StopIteration:
        print("все изображения считаны")

    except Exception:
        print("ЧТО - ТО СЛУЧИЛОСЬ ")

    finally:
        print("Закончил создание массива картинок ")


    diff_image=[]
    for i in range(len(images)):
        diff_image.append(images_water_mark[i]-image[i])


    for im in diff_image:
        im[im>100]=255


    for i in range(len(images)):
        images[i]=images[i]+diff_image[i]

    number_image=1

    for i in range(len(images)):
        img=Image.fromarray(images[i].astype(np.uint8))

        name_image = "diff" + str(number_image)

        path_save = rf"DiffImage/{name_image}.tif"

        if (os.path.exists(path_save)):
            os.remove(path_save)
        img.save(path_save)
        img.close()
        number_image += 1

def test(cv, res_cv):
    test = (cv == res_cv)
    return test

def MSE_image(x,y):
    e= (np.sum((x - y) ** 2)) / (len(x) * len(y[0]))
    return e

def psnr(W, Wr):
 e = (np.sum((W - Wr) ** 2)) / (len(W) * len(W[0]))
 p = 10 * np.log10(255 ** 2 / e)
 return p


def LWT2EmbedWaterMark(path_waterMark,path_dataSet,path_save_dir):
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














if __name__ == '__main__':
    path_waterMark="Water Mark Image/crown32.jpg"
    path_dataSet='DataSet'
    path_save_water_mark_image='CW'



    attack=Attack()
    attack.All_Attack()



    #WaterMark=LoadWaterMark.load(path_waterMark)
    #LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_water_mark_image)


    # path_dataSet_test='TestImage'
    # path_image_water_test='TestImageWaterMark'

    # extract_feature = ImageFeature(WaterMark)
    # path_save_feature_vec="feature_vec/image_water_mark_dataset.txt"
    # path_image_water_test='CW'
    # extract_feature.save_feature_data(path_save_feature_vec,path_image_water_test)
    # extract_feature.close()










