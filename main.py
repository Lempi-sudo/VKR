# import os
# from PIL import Image
# from matplotlib import pyplot as plt
# from skimage.io import imread
# import numpy as np
from pywt import wavedec
import matlab.engine
import numpy as np
from ImageWork import LoadNamesImage, LoadImage
from skimage.io import imread
import matlab
import pywt
from ImageWork import LoadNamesImage, LoadImage
from MatLabCalculation import LiftingWaveletTransform





if __name__ == '__main__':
    load_name = LoadNamesImage()
    path_name_image = load_name.get_list_image_name('TestDataSet')
    load_image = LoadImage(path_name_image)
    matlwt2=LiftingWaveletTransform()
    eng2 = matlab.engine.start_matlab()

    i=0

    try:
        while True:
            print(f"картинок обработано {i}")
            image = load_image.next_image()
            CA, CH, CV, CD = matlwt2.lwt2(image)

            cv=matlwt2.getCV_numpy()



            i+=1
    except StopIteration:
        print("все изображения считаны")
    except Exception:
        print("ЧТО - ТО СЛУЧИЛОСЬ ")
    finally:
        print("выкл matlab")
        matlwt2.exit_engine()

    print()
