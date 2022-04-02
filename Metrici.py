import numpy as np
from skimage.io import imread
import os
from WatermarkEmbedding import WatermarkEmbedding
from ImageWork import ImagesNamesLoader, ImageLoader, WaterMarkLoader
from MatLabCalculation import LiftingWaveletTransform, Transform_Matlab_to_NP
from PIL import Image



#Неточность может возникать в ilwt2 или в save()
def test_lwt2_before_after_save(path_waterMark): #плохие тесты почему то разные значения для казалось бы одинаковых преобразования
    water_mark = WaterMarkLoader.load(path_waterMark)  # считывание водяного знака
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

    load_name = ImagesNamesLoader()
    path_name_image = load_name.get_list_image_name('TestImagewithWaterMark')
    load_image = ImageLoader(path_name_image)
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



    load_name = ImagesNamesLoader()
    path_name_image = load_name.get_list_image_name('TestImage')
    load_image = ImageLoader(path_name_image)
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





