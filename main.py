# import os
# from PIL import Image
# from matplotlib import pyplot as plt
# from skimage.io import imread
# import numpy as np
from pywt import wavedec
import pywt
from ImageWork import LoadNamesImage, LoadImage





if __name__ == '__main__':
    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name('TestDataSet')
    load_image = LoadImage(path_name_image)

    try:
        while True:
            image = load_image.next_image()
            embed_wav = pywt.WaveletPacket2D(image, 'haar')
            h = embed_wav['h'].data.copy()
            a=embed_wav['a'].data.copy()
    except StopIteration:
        print("все изображения считаны")

    print()
