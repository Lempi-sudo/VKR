from Tasks import Task1, Task4 , Task7,  LWT2EmbedWaterMark , all_attack, all_feature
from AttackInImage import *


def embed_readble_wm():
    path_waterMark = "Water Mark Image/waterMark3.jpg"
    path_dataSet = 'DataSetTest'
    path_save_water_mark_image = 'ImgWMReadble'
    LWT2EmbedWaterMark(path_waterMark, path_dataSet, path_save_water_mark_image)


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


def cv2PSRN(W, Wr):
    p = cv2.PSNR(W, Wr)
    return p


# результат выдает побитовое сравнение в процентах насколько похожи два ЦВЗ
def pobitovo_sravnenie_WaterMark(W1, W2, total_bit=1024):
    t1 = W1 == W2
    sum = t1.sum()
    return sum / total_bit * 100


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


