from scipy.ndimage import median_filter
from ImageWork import ImagesNamesLoader, ImageLoader, WaterMarkLoader
import os
from PIL import Image ,ImageEnhance
import numpy as np
from skimage.util import random_noise
import cv2
from scipy.ndimage import gaussian_filter


class Attack:
    # хз как это должно работать , пока не использую
    def Crop(self, path_image, path_image_attacked, p):
        images_name_loader = ImagesNamesLoader()
        path_image_list = images_name_loader.get_image_name_list(path_image)
        image_loader = ImageLoader(path_image_list)

        number_image = 1
        i = 1

        try:
            while True:
                image = image_loader.next_image()
                width, height = image.shape
                image = Image.fromarray(image.astype(np.uint8))

                left = 0
                top = 0
                right = int(width * (1 - p))
                bottom = int(height * (1 - p))

                bad_image = image.crop((left, top, right, bottom))

                bridge_np = np.asarray(bad_image)

                name_image = "Crop" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                bad_image.save(path_save)
                bad_image.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано Crop {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    # НЕ РАБОТАЕТ
    def Gaussian_noise_attack(self, path_image, path_image_attacked, p=0.01):
        image_name_loader = ImagesNamesLoader()
        path_image_list = image_name_loader.get_image_name_list(path_image)
        image_loader = ImageLoader(path_image_list)

        number_image = 1
        i = 1

        try:
            while True:
                image = image_loader.next_image()

                bad_image_norm = random_noise(np.full(image.shape, -1), mode='gaussian', var=0.01)

                max = 255
                min = 0

                bad_image = bad_image_norm * 255

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

    # Работает но не понятно правильно ли ?
    def Gaussian_filter_attack(self, path_image, path_image_attacked, window=3):
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                kernel = 1 / 3

                bad_image = gaussian_filter(image, sigma=kernel)

                test = image - bad_image

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


    #Работает!!! , но уже не помню как :)=
    def median_attack(self, path_image, path_image_attacked, window=(3,3)):
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

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
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

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
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

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

    def Save_JPEG(self, path_image, path_image_attacked, QF):
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                img = Image.fromarray(image.astype(np.uint8))
                name_image = "Save_JPEG" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.jpeg"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save, quality=QF)
                img.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано jpeg сжатием {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    def Histogram(self, path_image, path_image_attacked):
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                # calculate hist
                hist, bins = np.histogram(image, 256)
                # calculate cdf
                cdf = hist.cumsum()
                # plot hist

                cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
                cdf = cdf.astype(np.uint8)  # Transform from float64 back to unit8

                bad_image = np.zeros((255, 255, 1), dtype=np.uint8)
                bad_image = cdf[image]




                img = Image.fromarray(bad_image.astype(np.uint8))
                name_image = "Histogram" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано histogram {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    def Gamma_Correction(self , path_image, path_image_attacked, gamma=1.6):
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()

                invGamma = 1.0 / gamma
                bad_image =(( image / 255 ) **invGamma ) *255
                bad_image=np.rint(bad_image)


                img = Image.fromarray(bad_image.astype(np.uint8))
                name_image = "GammaCorrection" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                img.save(path_save)
                img.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано Gamma {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")

    def Sharpness(self, path_image, path_image_attacked):
        load_name = ImagesNamesLoader()
        path_name_image = load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)

        number_image = 1
        i = 1

        try:
            while True:
                image = load_image.next_image()
                image = Image.fromarray(image.astype(np.uint8))
                enhancer = ImageEnhance.Sharpness(image)

                factor = 2
                bad_image = enhancer.enhance(factor)


                name_image = "Sharpness" + str(number_image)

                path_save = rf"{path_image_attacked}/{name_image}.tif"

                if (os.path.exists(path_save)):
                    os.remove(path_save)
                bad_image.save(path_save)
                bad_image.close()
                number_image += 1

                if (i % 25 == 0):
                    print(f"картинок атаковано Sharpness {i}")
                i += 1

        except StopIteration:
            print("все изображения считаны")




    def All_Attack(self):
        image_path='CW'
        image_attacked_salt_paper='AttackedImage/SaltPaperAttack'
        image_attacked_median='AttackedImage/medianAttack'
        image_attacked_average = 'AttackedImage/AverageAttack'
        image_save_jpeg20 = 'AttackedImage/JPEG20'
        image_save_jpeg30 = 'AttackedImage/JPEG30'
        image_save_jpeg40 = 'AttackedImage/JPEG40'
        image_save_jpeg50 = 'AttackedImage/JPEG50'
        image_histogram = 'AttackedImage/HistogramAttack'
        Gamma_Correction = 'AttackedImage/GammaCorrection'
        Sharpness = 'AttackedImage/Sharpness'

        self.median_attack(path_image=image_path , path_image_attacked=image_attacked_median )
        self.salt_peper_attack(path_image=image_path , path_image_attacked= image_attacked_salt_paper)
        self.average_filter(path_image=image_path, path_image_attacked=image_attacked_average)
        self.Save_JPEG(image_path, image_save_jpeg50, 50)
        self.Histogram(image_path,image_histogram)
        self.Gamma_Correction(image_path, Gamma_Correction)
        self.Sharpness(image_path, Sharpness)

