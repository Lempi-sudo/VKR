'''
Модуль для разных вспомогательных функций
'''

import numpy as np
from PIL import Image



class GenerateBinaryWaterMark:
    @staticmethod
    def water_mark(size=32 * 32):
        return np.random.randint(2, size=size)



def GenerateWaterMark(IsSave = False, path_save = rf"Water Mark Image/WaterMarkRandom.tif"):
    '''
    Функция генерирует случайный набор из множества {0,1}
    C Равномерным распределением.
    В виде квадрата (32,32)
    и сохраняет по пути path_save ,если установлен флаг IsSave
    '''
    wm = np.random.randint(0, 2, (32, 32))
    wm_res=wm.copy()
    if IsSave:
        wm[wm > 0] = 255
        img = Image.fromarray(wm.astype(np.uint8))
        img.save(path_save)
        img.close()
    return wm_res


