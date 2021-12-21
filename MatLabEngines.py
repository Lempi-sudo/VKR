import matlab.engine
import numpy as np
from ImageWork import LoadNamesImage, LoadImage
from skimage.io import imread
import matlab

if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    path = 'D:/Code/MatLab/NIR/Image00001.tif'
    image = eng.double(eng.imread(path))

    try:

        [CA, CH, CV, CD] = eng.lwt2(image, 'haar', 3, nargout=4)

        ca = np.array(CA)

        mat_a = matlab.double(ca.tolist())

        sourse_image = eng.ilwt2(mat_a, CH, CV, CD, 'haar', 3, nargout=1)

        s_im = np.array(sourse_image)

    except Exception:
        eng.exit()

    finally:
        eng.exit()
