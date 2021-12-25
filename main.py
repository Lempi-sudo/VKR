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

def LWT2EmbedWaterMark(path_waterMark,path_dataSet,path_save_dir):
    water_mark = LoadWaterMark.load(path_waterMark)  # считывание водяного знака

    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name(path_dataSet)

    load_image = LoadImage(path_name_image)

    mat_lab_lwt2 = LiftingWaveletTransform()
    scheme_embedding = WatermarkEmbedding(water_mark)
    transformator = Transform_Matlab_to_NP()

    i = 0
    number_image = 0

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





if __name__ == '__main__':
    path_waterMark="Water Mark Image/crown32.jpg"
    path_dataSet='TestDataSet'
    path_save_dir='Image With WaterMark'

    LWT2EmbedWaterMark(path_waterMark,path_dataSet,path_save_dir)












