from Tasks import Task1, Task2


if __name__ == '__main__':
    path_waterMark = "Water Mark Image/WhiteWaterMark3.jpg" # путь до водяного знака3 (Дота2)
    path_dataSet = 'DataSet' # путь до набора картинок
    path_save_CW = 'Task2CW'
    path_feature_vec= "feature_vec/Task2/Task2FeatureVec.txt"

    Task2()



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


