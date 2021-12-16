import numpy as np
import pywt
import os
from skimage import io
from PIL import Image

s = 64
def psnr(W, Wr):
    e = (np.sum((W - Wr) ** 2)) / (len(W) * len(W[0]))
    p = 10 * np.log10(255 ** 2 / e)
    return p

def generationOmega(size):
    size_cvz = size * size
    omega = np.random.normal(loc=0.0, scale=1.0, size=int(size_cvz))
    return omega

def embedding_info (space, omega_matrica, alfa):
    # CW = np.mean(space) + (space - np.mean(space)) * (1 + alfa * omega_matrica) #mult

    int_matrix = (space/128).astype(int)
    CW = space + omega_matrica * alfa * ((-1) ** int_matrix)
    return CW

def omegaTilda (f_Wm, f_m, alfa):
    int_matrix = (f_m/128).astype(int)
    omegaT = (f_Wm-f_m) / (alfa * ((-1)**int_matrix))

    # omegaT = (f_Wm - f_m) / (alfa * (f_m - np.mean(f_m))) #mult
    return omegaT

def testRO (Omega, Omega_Tilda):
    return (np.sum(Omega * Omega_Tilda)) / ((np.sqrt(np.sum(Omega ** 2))) * (np.sqrt(np.sum(Omega_Tilda ** 2))))

if __name__ == '__main__':

    #1 генерация ЦВЗ - псевдослучайная последовательность
    omega = generationOmega(s)
    psnr_max = 30
    omega_matrica = omega.reshape(s, s) #Встраиваимый спектр
    AlfaList = np.arange(0.01, 30, 0.3, dtype=float)
    for alfa in AlfaList:
        #2 трансформация исходного изображения
        bridge_image = io.imread(r"bridge.tif")
        bridge_np = np.asarray(bridge_image)
        wav = pywt.WaveletPacket2D(bridge_np, 'haar')
        feature_space = wav['aaa'].data.copy()  # LLL
        f_contener = np.ravel(feature_space)

        #3 встраивание информации аддитивным методом
        # alfa = 1 # пока произвольное
        embed_space = embedding_info(feature_space, omega_matrica, alfa)

        # #4 сформирование носителя информации (обратное преобразование)
        wav['aaa'].data = embed_space
        image_embed = wav.reconstruct(update=False)
        image_embed = Image.fromarray(image_embed.astype('uint8'))
        if (os.path.exists(r"image_embed.tif")):
            os.remove(r"image_embed.tif")
        image_embed.save(r'image_embed.tif')
        image_embed.close()

        #5 считыывание носителя информации
        embed_image = io.imread(r"image_embed.tif")
        embed_image_np = np.asarray(embed_image)
        embed_wav = pywt.WaveletPacket2D(embed_image_np, 'haar')
        embed_feature_space = embed_wav['aaa'].data.copy()  # LLL
        # f_wTilda = np.ravel(embed_feature_space)

        # 6 оценка встрренного изображения
        omega_Tilda = omegaTilda(embed_feature_space, feature_space, alfa)
        # omega_wav = pywt.WaveletPacket2D(omega_matrica, 'haar')
        # omega_wav = omega_wav.data.copy()
        # omega_wav = np.ravel(omega_wav)

        ro = testRO(omega, np.ravel(omega_Tilda))
        # print("alfa=%.1f" % alfa, "ro=%.3f" % ro, "psnr=%.3f" % psnr(bridge_np, embed_image_np))

        if ro > 0.5:

            if psnr(bridge_np, embed_image_np) > psnr_max:
            # print('daaa')
                print("alfa=%.2f" % alfa, "ro=%.3f" % ro, "psnr=%.3f" % psnr(bridge_np, embed_image_np))
                psnr_max = psnr(bridge_np, embed_image_np)

    #7 подбор alfa
