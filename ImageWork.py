import os
from skimage.io import imread


class LoadNamesImage:
    def __init__(self):
        self.countImage = 0

    def get_list_image_name(self, directory_name: str):
        image_name_list = []
        for dirpath, dirnames, filenames in os.walk(directory_name):
            for filename in filenames:
                path = dirpath + '/' + filename
                image_name_list.append(path)
                self.countImage += 1
            return image_name_list

class LoadImage:
    def __init__(self, name_image_list: list):
        self.name_image_list = name_image_list
        self.cont_image = len(name_image_list)
        self.iterListImage = iter(name_image_list)

    def next_image(self):
        if self.cont_image > 0:
            name_path_image = next(self.iterListImage)
            image_numpy = imread(name_path_image)
            self.cont_image -= 1
        else:
            raise StopIteration
        return image_numpy

class LoadWaterMark():

    @staticmethod
    def load(path , treshold=180):
        water_mark = imread(path)
        water_mark[water_mark < treshold] = 0
        water_mark[water_mark >= treshold] = 1
        if water_mark.shape[0] > 32:
            water_mark = water_mark[20:52, 31:63]


        water_markr_res = water_mark.ravel()



        return water_markr_res

