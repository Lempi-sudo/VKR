import matlab.engine
import numpy as np
from ImageWork import LoadNamesImage, LoadImage





if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    path = 'D:/Code/MatLab/NIR/Image00001.tif';
    image = eng.double(eng.imread(path));
    X_InPlace3 = eng.lwt2(image, 'haar', 3 )

    [CA,CH,CV,CD] = eng.lwt2(image,'haar',3,nargout=4)

    ca=np.array(CA)

    sourse_image = eng.ilwt2(CA,CH,CV,CD, 'haar', 3 , nargout=1)

    s_im = np.array(sourse_image)

    eng.exit()
