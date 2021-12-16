import random
import scipy.signal
from scipy import interpolate, signal
import pywt
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.io import imread
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

def embedding_MSE (img, embed):
    n = 50
    kernel = np.ones((n, n), dtype=float) / n ** 2
    m2x = scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm') ** 2
    mx2 = scipy.signal.convolve2d(img.astype(int) ** 2, kernel, mode='same', boundary='symm')
    tmp = mx2 - m2x
    tmp[tmp<0] = 0
    std = np.sqrt(tmp)
    # plt.imshow(std, cmap='gray')
    # plt.show()

    beta = std / std.max()
    CW_mse = embed * beta + img * (1 - beta)
    # plt.imshow(CW_mse, cmap='gray')
    # plt.show()

    return CW_mse

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

def testTrueRO ():
    res_RO = 0
    res_alfa = 0
    #1 генерация ЦВЗ - псевдослучайная последовательность
    omega = generationOmega(s)
    psnr_max = 30
    omega_matrica = omega.reshape(s, s) #Встраиваимый спектр
    AlfaList = np.arange(0.01, 10, 0.3, dtype=float)
    for alfa in AlfaList:
        #2 трансформация исходного изображения
        bridge_image = imread("barb.tif")
        bridge_np = np.asarray(bridge_image)
        wav = pywt.WaveletPacket2D(bridge_np, 'haar')
        feature_space = wav['aaa'].data.copy()  # LLL

        #3 встраивание информации аддитивным методом
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
        embed_image = imread(r"image_embed.tif")
        embed_image_np = np.asarray(embed_image)
        embed_wav = pywt.WaveletPacket2D(embed_image_np, 'haar')
        embed_feature_space = embed_wav['aaa'].data.copy()  # LLL

        # 6 оценка встрренного изображения
        omega_Tilda = omegaTilda(embed_feature_space, feature_space, alfa)
        ro = testRO(omega, np.ravel(omega_Tilda))
        # print("alfa=%.1f" % alfa, "ro=%.3f" % ro, "psnr=%.3f" % psnr(bridge_np, embed_image_np))

        if ro > res_RO:

            if psnr(bridge_np, embed_image_np) > 30:
            # print('daaa')
            #     print("alfa=%.2f" % alfa, "ro=%.3f" % ro, "psnr=%.3f" % psnr(bridge_np, embed_image_np))
                psnr_max = psnr(bridge_np, embed_image_np)
            res_RO = ro
        res_alfa = alfa
    return res_RO, omega, res_alfa

def cyclic_shift(image, roll_f):
    shifted_image = image.copy()
    height, width = shifted_image.shape
    shifted_image = np.roll(shifted_image, int(roll_f * height), axis=0)
    shifted_image = np.roll(shifted_image, int(roll_f * width), axis=1)
    return shifted_image

def solt_and_paper(image, q):
    result_image = image.copy()
    height, width = image.shape

    def calculation (image, pixels, epsilon, height, width):
        for i in range(pixels):
            rand_x = random.randint(0, height-1)
            rand_y = random.randint(0, width-1)
            image[rand_x][rand_y] += epsilon

            if image[rand_x][rand_y] > 255:
                image[rand_x][rand_y] = 255
            if image[rand_x][rand_y] < 0:
                image[rand_x][rand_y] = 0
        return image

    pixels_for_salt_proportion = round(height * width * q/2)
    pixels_for_paper_proportion = round(height * width * (1-q))

    result_image = calculation(result_image, pixels_for_salt_proportion, 255, height, width)
    result_image = calculation(result_image, pixels_for_salt_proportion, -255, height, width)
    result_image = calculation(result_image, pixels_for_paper_proportion, 0, height, width)

    return result_image

def scale_rest (image, p):
    result_image = image.copy()
    height, width = image.shape

    xrange = lambda x: np.linspace(0, 1, x)
    f = interpolate.interp2d(xrange(height), xrange(width), result_image, kind="linear")
    result_image = f(xrange(int(p * height)), xrange(int(p * width)))

    h_new, w_new = result_image.shape
    f = interpolate.interp2d(xrange(h_new), xrange(w_new), result_image, kind="linear")
    result_image = f(xrange(height), xrange(width))
    return result_image

def smooth (image, m):
    result_image = image.copy()
    g = (1 / m / m) * np.ones((m, m))
    result_image = (signal.convolve2d(result_image, g, mode='same', boundary='symm'))
    return result_image

def process_distortion(distortion_func, min, max, delta, omega, alfa):
    # alfa = 9.91
    # omega = generationOmega(s)
    omega_matrica = omega.reshape(s, s)
    img = imread("barb.tif")
    img_np = np.asarray(img)
    wav = pywt.WaveletPacket2D(img_np, 'haar')
    feature_space = wav['aaa'].data.copy()
    embed_space = embedding_info(feature_space, omega_matrica, alfa)

    wav['aaa'].data = embed_space
    image_embed = wav.reconstruct(update=False)
    # image_embed = Image.fromarray(image_embed.astype('uint8'))

    params = np.arange(min, max+delta, delta)
    proximities = []
    PSNR = []
    for param in params:
        distorted_img = distortion_func(image_embed, param)

        image_save = Image.fromarray(distorted_img.astype('uint8'))
        image_save.save('distor_img.tif')

        distorted_wav = pywt.WaveletPacket2D(distorted_img, 'haar')
        embed_feature_space = distorted_wav['aaa'].data.copy()
        omega_Tilda = omegaTilda(embed_feature_space, feature_space, alfa)

        proximities.append(testRO(omega, np.ravel(omega_Tilda)))
        PSNR.append(psnr(img, distorted_img))
    print(proximities)
    print(PSNR)

    fig, axs = plt.subplots(2)
    fig.suptitle(f'Результаты для {distortion_func.__name__}')

    axs[0].plot(params, proximities)
    axs[0].set(ylabel='Значения функции близости')
    axs[1].plot(params, PSNR)
    axs[1].set(ylabel='PSNR')

    plt.savefig('No_MSE')

def process_distortion_MSE (distortion_func, min, max, delta, omega, alfa):
    omega_matrica = omega.reshape(s, s)
    img = imread("barb.tif")
    img_np = np.asarray(img)
    wav = pywt.WaveletPacket2D(img_np, 'haar')
    feature_space = wav['aaa'].data.copy()
    embed_space = embedding_info(feature_space, omega_matrica, alfa)

    wav['aaa'].data = embed_space
    image_embed = wav.reconstruct(update=False)

    image_embed = embedding_MSE(img, image_embed)

    params = np.arange(min, max + delta, delta)
    proximities = []
    PSNR = []
    for param in params:
        distorted_img = distortion_func(image_embed, param)

        image_save = Image.fromarray(distorted_img.astype('uint8'))
        image_save.save('distor_img_MSE.tif')

        distorted_wav = pywt.WaveletPacket2D(distorted_img, 'haar')
        embed_feature_space = distorted_wav['aaa'].data.copy()
        omega_Tilda = omegaTilda(embed_feature_space, feature_space, alfa)

        proximities.append(testRO(omega, np.ravel(omega_Tilda)))
        PSNR.append(psnr(img, distorted_img))
    print(proximities)
    print(PSNR)

    fig, axs = plt.subplots(2)
    fig.suptitle(f'Результаты для {distortion_func.__name__}')

    axs[0].plot(params, proximities)
    axs[0].set(ylabel='Значения функции близости')
    axs[1].plot(params, PSNR)
    axs[1].set(ylabel='PSNR')

    plt.savefig('MSE')

if __name__ == '__main__':
    res_ro, omega, res_alfa = testTrueRO()
    print(res_ro)
    print(omega)
    print(res_alfa)
    process_distortion(cyclic_shift, 0.1, 0.9, 0.1, omega, res_alfa)
    # process_distortion(solt_and_paper, 0.05, 0.5, 0.05, omega, res_alfa)
    # process_distortion(scale_rest, 0.55, 1.45, 0.15, omega, res_alfa)
    # process_distortion(smooth, 3, 15, 2, omega, res_alfa)

    process_distortion_MSE(cyclic_shift, 0.1, 0.9, 0.1, omega, res_alfa)
    # process_distortion_MSE(solt_and_paper, 0.05, 0.5, 0.05, omega, res_alfa)
    # process_distortion_MSE(scale_rest, 0.55, 1.45, 0.15, omega, res_alfa)
    # process_distortion_MSE(smooth, 3, 15, 2, omega, res_alfa)

