import os
from PIL import Image
from ImageWork import LoadNamesImage, LoadImage
from matplotlib import pyplot as plt
from skimage.io import imread
import numpy as np


# def test():
#     listdir = os.listdir()
#
#     tmpdir = os.getcwd()
#
#     for dirpath, dirnames, filenames in os.walk("TestDataSet"):
#         # перебрать каталоги
#         for dirname in dirnames:
#             print("Каталог:", os.path.join(dirpath, dirname))
#         # перебрать файлы
#         for filename in filenames:
#             path = dirpath + '/' + filename
#             embed_image = imread(path)
#             img = Image.fromarray(embed_image.astype(np.uint8))
#             Image_RGB_np2 = np.asarray(img)
#             image = Image.open(path)
#             Image_RGB_np = np.asarray(image)
#             image.show()
#             plt.imshow(image, cmap='grey')
#             plt.show()
#             print("Файл:", os.path.join(dirpath, filename))
#
#     print("Все папки и файлы:", os.listdir())


if __name__ == '__main__':
    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name('TestDataSet')
    load_image = LoadImage(path_name_image)

    try:
        while True:
            image = load_image.next_image()
            print(image)

    except StopIteration:
        print("все изображения считаны")

    print()
