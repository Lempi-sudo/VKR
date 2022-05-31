from ImageWork import ImagesNamesLoader, ImageLoader
import numpy as np
import pandas as pd
from MatLabCalculation import LiftingWaveletTransform, Transform_Matlab_to_NP

class ImageFeature:
    '''
    Класс который реализует формирование вектора признаков из изображения
    '''

    def __init__(self, water_mark):
        '''
        Инициализация всех необходимых объектов
        :param water_mark: матрица значений пикелей Водяного знака
        '''
        self.water_mark = water_mark
        self.mat_lab = LiftingWaveletTransform()
        self.load_name = ImagesNamesLoader()
        self.transformator = Transform_Matlab_to_NP()

    def __extract_sample_block__(self, f):
        '''
        формирует список блоков из учета того что водяной знак встраивали во всё изображение
        :param f: Всё изображение
        :return: список блоков всего изображения
        '''
        size = f.shape[0]
        blocks = []
        pse=(2,size,8)
        for i in range(pse[0],pse[1],pse[2]):
            for j in range(pse[0],pse[1],pse[2]):
                block = f[i:i + 2, j:j + 2]
                blocks.append(block)
        return blocks

    def __extract_hl2_area__(self, CV_Block, border_intex=(128, 192, 64, 128)):
        '''
        Получаю блок CV ( это нижний левый квадрат 256х256 ) и извлекаю оттуда квадрат hl2
        :param CV_Block: Матрица 256х256 , получённая после lwt  преобразования третьего уровня.
        :param border_intex: Значения пикселей между которыми нуобходимо извлекать блок
        :return: Матрица 64х64
        '''
        hl2 = CV_Block[border_intex[0]:border_intex[1], border_intex[2]:border_intex[3]]
        return hl2

    def __all_blocks__(self, hl2):
        '''
        Получаю блок HL2 (64х64) и разбиваю на 1024 блока (2х2)
        :param hl2: это частью значений частотного спектра изображения в которое происходит встраивание информации
        :return: список матриц 2х2
        '''
        size = hl2.shape[0]
        blocks = []
        split = [x for x in range(2, size, 2)]
        list_split = np.array_split(hl2, indices_or_sections=split, axis=0)
        for spl in list_split:
            block = np.split(spl, indices_or_sections=split, axis=1)
            blocks.extend(block)
        return blocks



    # CЛАБОЕ МЕСТО ДОЛГО ФОРМИРУЕТСЯ ВЕКТОР ПРИЗНАКОВ
    # ПОПРОБОВАТЬ УБРАТЬ ФОРМИРОВАНИЕ И СОХРАНЕНИЕ ВЕКТОРА ПРИЗНАКОВ,
    # ОСТАВИВ ТОЛЬКО ДОЛГУЮ ЧАТЬ LWT2 ПРЕОБРАЗОВАНИЯ
    # И СРАВНИТЬ СКОРОСТЬ РАБОТЫ
    def save_feature_data_hl2(self, path_save, path_image):
        '''
        Данная функция формирует вектор признаков из всех доступных изображений в директории  path_image
        и сохраняет в path_save в виде txt файла с таблицей csv(разделитель запятая)
        :param path_save:путь для сохранения документа(txt) с вектором признаков
        :param path_image:путь до изображений из которых формируют вектор признаков
        '''
        path_name_image = self.load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)
        i = 1
        listrow = []
        try:
            while True:
                image = load_image.next_image()
                CA, CH, CV, CD = self.mat_lab.lwt2(image , level=1)
                c = self.transformator.get_NP(CV)
                hl2 = self.__extract_hl2_area__(c)
                blocks = self.__all_blocks__(hl2)
                iter_water = iter(self.water_mark)
                for block in blocks:
                    vec = np.ravel(block)
                    new_row = {'bit': next(iter_water), "feature_1": vec[0], "feature_2": vec[1], "feature_3": vec[2],
                               "feature_4": vec[3]}
                    listrow.append(new_row)
                print(f"Векторов признаков сформировано {i}")
                i += 1
        except StopIteration:
            print("Из всех изображений сформированы вектора признаков")
        columns = ['bit', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
        df = pd.DataFrame(data=listrow, columns=columns)
        df.to_csv(path_save)

    def save_all_image_feature_data(self, path_save, path_image):
        '''
        Данная функция формирует вектор признаков из всех доступных изображений в директории  path_image
        и сохраняет в path_save в виде txt файла с таблицей csv(разделитель запятая)
        :param path_save:путь для сохранения документа(txt) с вектором признаков
        :param path_image:путь до изображений из которых формируют вектор признаков
        РАБОТАЕТ ДЛЯ ВСЕГО ИЗОБРАЖЕНИЯ
        '''
        path_name_image = self.load_name.get_image_name_list(path_image)
        load_image = ImageLoader(path_name_image)
        i = 1
        listrow = []
        try:
            while True:
                image = load_image.next_image()
                CA, CH, CV, CD = self.mat_lab.lwt2(image, level=1)
                c = self.transformator.get_NP(CH)
                blocks = self.__extract_sample_block__(c)
                iter_water = iter(self.water_mark)
                for block in blocks:
                    vec = np.ravel(block)
                    new_row = {'bit': next(iter_water), "feature_1": vec[0], "feature_2": vec[1],
                               "feature_3": vec[2],
                               "feature_4": vec[3]}
                    listrow.append(new_row)
                print(f"Векторов признаков сформировано {i}")
                i += 1
        except StopIteration:
            print("Из всех изображений сформированы вектора признаков")
        columns = ['bit', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
        df = pd.DataFrame(data=listrow, columns=columns)
        df.to_csv(path_save)


    def close(self):
        '''
        процедура отключения движка MatLab
        '''
        self.mat_lab.exit_engine()
