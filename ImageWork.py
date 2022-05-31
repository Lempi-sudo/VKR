import os
from skimage.io import imread
from ExceptionsModule import WaterMarkWrong


class ImagesNamesLoader:
    def __init__(self):
        self.countImage = 0

    def get_image_name_list(self, directory_name: str):
        image_name_list = []
        for dirpath, dirnames, filenames in os.walk(directory_name):
            for filename in filenames:
                path = dirpath + '/' + filename
                image_name_list.append(path)
                self.countImage += 1
            return image_name_list


class ImageLoader:
    def __init__(self, name_image_list: list):
        self.name_image_list = name_image_list
        self.cont_image = len(name_image_list)
        self.name_images_iterator = iter(name_image_list)

    def next_image(self):
        if self.cont_image > 0:
            name_path_image = next(self.name_images_iterator)
            image_numpy = imread(name_path_image)
            self.cont_image -= 1
        else:
            raise StopIteration
        return image_numpy


class WaterMarkLoader:
    @staticmethod
    def load(path, threshold=100, right_size_watermark=32):
        try:
            water_mark = imread(path)
            if water_mark.shape[0] != right_size_watermark or water_mark.shape[1] != right_size_watermark:
                raise WaterMarkWrong("Неправильный  водяной знак", right_size_watermark, water_mark.shape[0],
                                     water_mark.shape[1])
            if len(water_mark.shape)==3:
                if water_mark.shape[2] != 1:
                    water_mark = water_mark[:, :, 0]
                    water_mark[water_mark < threshold] = 0
                    water_mark[water_mark >= threshold] = 1
            else:
                water_mark[water_mark < threshold] = 0
                water_mark[water_mark >= threshold] = 1
        except WaterMarkWrong as e:
            print(e)
            raise SystemExit(1)

        return water_mark.ravel()
