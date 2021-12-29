from scipy.ndimage import median_filter
from ImageWork import LoadNamesImage, LoadImage, LoadWaterMark
import os
from PIL import Image
import numpy as np
from skimage.util import random_noise
import cv2
from scipy.ndimage import gaussian_filter






class Attack:

    def median_attack(self, path_image, path_image_attacked, window=(3,3)):
        load_name = LoadNamesImage()
        path_name_image = load_name.get_list_image_name(path_image)
        load_image = LoadImage(path_name_image)

        number_image = 1
        i = 1

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

                if i % 25 == 0:
                    print(f"картинок атаковано медианным фильтром {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    def salt_peper_attack(self, path_image, path_image_attacked, p=0.01):
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

                bad_image[noise == -1] = 0
                bad_image[noise == 1] = 255

                img = Image.fromarray(bad_image.astype(np.uint8))
                name_image = "SaltPaperAttackedImage" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано соль-перец {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    def average_filter(self,path_image, path_image_attacked, window=3):
        load_name = LoadNamesImage()
        path_name_image = load_name.get_list_image_name(path_image)
        load_image = LoadImage(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                kernel = np.ones((window, window), np.float32) / (window*window)
                bad_image = cv2.filter2D(image, -1, kernel)

                img = Image.fromarray(bad_image.astype(np.uint8))
                name_image = "AverageAttack" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано Средний фильтр  {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    #НЕ РАБОТАЕТ
    def Gaussian_noise_attack(self, path_image, path_image_attacked, p=0.01):
        load_name = LoadNamesImage()
        path_name_image = load_name.get_list_image_name(path_image)
        load_image = LoadImage(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                bad_image_norm = random_noise(np.full(image.shape, -1), mode='gaussian', var=0.01)

                max=255
                min=0

                bad_image=bad_image_norm*255


                img = Image.fromarray(bad_image.astype(np.uint8))
                name_image = "GaussianAttack" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано Гауса шум  {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")
    #Работает но не понятно правильно ли ?
    def Gaussian_filter_attack(self, path_image, path_image_attacked, window=3):
        load_name = LoadNamesImage()
        path_name_image = load_name.get_list_image_name(path_image)
        load_image = LoadImage(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                kernel = 1/3

                bad_image = gaussian_filter(image , sigma=kernel)

                test=image-bad_image

                img = Image.fromarray(bad_image.astype(np.uint8))
                name_image = "GaussianFilterAttack" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано GaussianFilterAttack  {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    def All_Attack(self):
        image_path='CW'
        image_attacked_salt_paper='AttackedImage/SaltPaperAttack'
        image_attacked_median='AttackedImage/medianAttack'
        image_attacked_average = 'AttackedImage/AverageAttack'
        image_Gaussian_Filter_attack = 'AttackedImage/GaussianFilterАttack'

        self.median_attack(path_image=image_path , path_image_attacked=image_attacked_median )
        self.salt_peper_attack(path_image=image_path , path_image_attacked= image_attacked_salt_paper)
        self.average_filter(path_image=image_path, path_image_attacked=image_attacked_average)
        #self.Gaussian_attack(path_image=image_path, path_image_attacked=image_Gaussian_attack) НЕ РАБОТАЕТ
