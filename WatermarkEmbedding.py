import numpy as np


class GenerateBinaryWaterMark:
    @staticmethod
    def water_mark(size=32 * 32):
        return np.random.randint(2, size=size)


class WatermarkEmbedding:
    def __init__(self, w):
        self.W = w


    def __extract_hl2_area__(self ,f):
        Nf=f.shape[0]
        lh1=f[Nf//2:Nf,0:Nf//2]

        Nlh1=lh1.shape[0]

        hl2=lh1[0:Nlh1//2,Nlh1//2:Nlh1]
        return hl2


    def embed(self,f):
        if f.shape[0]>64 or f.shape[1]>64:
            hl2 = self.__extract_hl2_area__(f)

        print(f)



if __name__ == '__main__':
    mat=np.random.randint(0, 5, size=(256,256))

    emb= WatermarkEmbedding(GenerateBinaryWaterMark.water_mark())

    emb.embed(mat)

