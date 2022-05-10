import numpy as np
from skimage.io import imread


class WatermarkEmbedding:
    '''
    В данном Класе реализуются методы вставки (внедрения) ЦВЗ
    вставка может осуществляться разными алгоритмами
    Пока реализован только основной алгоритм встраивания (из научной статьи).
    Тройное разлжение виевлет -> Разбиение на области -> Использование блоков 2x2
    Основной метод embed
    '''

    def __init__(self, w, T=3.46):
        '''
        :param w: Водяной знак. Квадрат (32х32) состоит из множества {0,1}.
        :param T: Пороговое значение. Используется в формуле встраивания в блок 2х2
        '''
        self.W = w
        self.T = T
        print("SVI c порогом = ", T)



    def __extract_hl2_area__(self, f, intex=(128, 192, 64, 128)):
        hl2 = f[intex[0]:intex[1], intex[2]:intex[3]]
        return hl2

    def __insert_hl2_inF__(self,f,hl2, intex=(128, 192, 64, 128)):
        f[intex[0]:intex[1], intex[2]:intex[3]]=hl2
        return f

    def __find_large_and_second_large_coefficient__(self, block):
        first_c = block.max()
        secont_c = block.min()
        block=block.ravel()
        for c in block:
            if c>secont_c and c != first_c:
                secont_c = c

        return first_c, secont_c

    def __calculate_G__(self, all_blocks, count_block=1024):
        sum_e_b_max = 0
        for block in all_blocks:
            f_c, s_c = self.__find_large_and_second_large_coefficient__(block)
            e_b_max = f_c - s_c
            sum_e_b_max += e_b_max

        G = sum_e_b_max / count_block
        return G

    def __water_mark_bit_embed_in_sample_block__(self, block, bit, G):
        f_c, s_c = self.__find_large_and_second_large_coefficient__(block)
        eb_max = f_c - s_c

        if bit == 1:
            if eb_max < max(G, self.T):
                f_c = f_c + self.T / 2
            else:
                f_c = f_c - self.T / 2
                print(rf"eb_max={eb_max} > max( {G}, {self.T})")

        else:
            f_c = f_c - eb_max

        block= block.astype(np.float)

        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                if(block[i,j]==block.max()):
                    block[i,j]=f_c
                    break



        return block

    def __all_blocks__(self, hl2):
        size = hl2.shape[0]
        blocks = []
        for i in range(0, size, 2):
            for j in range(0, size, 2):
                block = hl2[i:i + 2, j:j + 2]
                blocks.append(block)
        return blocks

    def embed(self, f):
        iter_water = iter(self.W)
        if f.shape[0] > 64 or f.shape[1] > 64:
            hl2 = self.__extract_hl2_area__(f)

        hl2_res=hl2.copy()
        hl2_res=hl2_res.astype(np.float)


        blocks = self.__all_blocks__(hl2)
        G = self.__calculate_G__(blocks)

        size = hl2.shape[0]
        # number_bit=0
        for i in range(0, size, 2):
            for j in range(0, size, 2):
                block = hl2[i:i + 2, j:j + 2]
                try:
                    bit = next(iter_water)
                except StopIteration:
                    print("кончился водяной знак")
                # print(f"бит водяного знака {bit} номер {number_bit}")
                # number_bit+=1
                block_with_water_mark = self.__water_mark_bit_embed_in_sample_block__(block, bit, G)

                hl2_res[i:i + 2, j:j + 2]=block_with_water_mark

        f_res=self.__insert_hl2_inF__(f,hl2_res)

        return  f_res
