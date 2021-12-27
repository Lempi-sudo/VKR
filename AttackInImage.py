from scipy.ndimage import median_filter
from ImageWork import LoadNamesImage, LoadImage, LoadWaterMark
import os
from PIL import Image
import numpy as np


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





