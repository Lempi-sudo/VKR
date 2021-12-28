from scipy.ndimage import median_filter
from ImageWork import LoadNamesImage, LoadImage, LoadWaterMark
import os
from PIL import Image
import numpy as np
from skimage.util import random_noise


class Attack:

    def median_attack(self, path_image, path_image_attacked,window=3):
        load_name = LoadNamesImage()
        path_name_image = load_name.get_list_image_name(path_image)
        load_image = LoadImage(path_name_image)

        number_image=1
        i=1

        try:
            while True:
                image = load_image.next_image()

                median_image = median_filter(image, size=window)

                img = Image.fromarray(median_image.astype(np.uint8))
                name_image = "MedianAttackedImage" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                print(f"картинок атаковано медианным фильтром {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")


    def salt_peper_attack(self, path_image, path_image_attacked, p=0.1):
        load_name = LoadNamesImage()
        path_name_image = load_name.get_list_image_name(path_image)
        load_image = LoadImage(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                noise = random_noise(np.full(image.shape, -1), mode='s&p', amount=p)

                bad_image = image.copy()

                for index, x in np.ndenumerate(image):
                    if noise[index] == -1:
                        bad_image[index] = 0
                    if noise[index] == 1:
                        bad_image[index] = 255

                img = Image.fromarray(bad_image.astype(np.uint8))
                name_image = "SaltPaperAttackedImage" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                print(f"картинок атаковано соль-перец {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")







