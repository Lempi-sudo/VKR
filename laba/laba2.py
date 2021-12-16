import numpy as np
from matplotlib import pyplot as plt
import os
from skimage import io
from PIL import Image, ImageDraw
from skimage.util import random_noise
from skimage.io import imread, imshow, show, imsave
from skimage.exposure import histogram, equalize_hist
from skimage import color
from scipy import fftpack
from time import sleep
from termcolor import colored, cprint


def psnr(W, Wr):
 e = (np.sum((W - Wr) ** 2)) / (len(W) * len(W[0]))
 p = 10 * np.log10(255 ** 2 / e)
 return p

def OmegaTilda(fW_tilda ,f ,a=1):

    return np.where(f!=0 ,((fW_tilda-f) / (a*f)),255)

def test_ro(Q , QTilda): #  неправильно переделать
    Q1=Q*QTilda
    t1=np.sum(Q1)
    t2=np.sqrt(np.sum(Q*Q))
    t3=np.sqrt(np.sum(QTilda*QTilda))
    t4=t3*t2
    res =t1/t4
    return  res#  неправильно переделать

def generationOmega(size,C,H):
    size_cvz = size * size
    CH = C * H
    size_omega = size_cvz * CH
    omega = np.random.normal(loc=0.0, scale=1.0,size=int(size_omega))
    return omega

if __name__ == '__main__':
    # длины из чисел, распределённых по нормаль-ному закону.
    omega = generationOmega(512, 1/4 , 1/2)
    omega_matrica = omega.reshape(256,128)
    AlfaList = np.arange(3 ,3.01 ,0.1 , dtype=float)

    for Alfa in AlfaList:
        bridge_image = Image.open(r"bridge.tif")
        bridge_np = np.asarray(bridge_image)
        fft_bridge = fftpack.fft(bridge_np)

        fft_copy=fft_bridge.copy()
        fft_bringe_imag=fft_copy.imag


        Hight_image=512
        Witht_image=512

        Witht_signal=128
        Hight_signal=256
        h_l_board = (Hight_image // 2) - (Hight_signal // 2)
        h_r_board = (Hight_image // 2) + (Hight_signal // 2)
        w_l_board = (Witht_image // 2) - (Witht_signal// 2)
        w_r_board = (Witht_image // 2) + (Witht_signal // 2)

        fft_bringe_imag[h_l_board:h_r_board, w_l_board:w_r_board] = fft_bringe_imag[h_l_board:h_r_board, w_l_board:w_r_board] * (1+Alfa * omega_matrica)
        fft_copy.imag=fft_bringe_imag
        real_image = fftpack.ifft(fft_copy)


        # for i in range(0, 128):
        #     fft_bringe_imag[128 + i, 256 - i:257 + i] = fft_copy.imag[128 + i, 256 - i:257 + i] * \
        #                                                  (1 + Alfa * omega[i * i:(i + 1) * (i + 1)])
        #
        # for i in range(0, 128):
        #     fft_bringe_imag[384 - i, 256 - i:257 + i] = fft_copy.imag[384 - i, 256 - i:257 + i] * \
        #                                                  (1 + Alfa * omega[
        #                                                              128 * 128 + i * i: 128 * 128 + (i + 1) * (i + 1)])

        fft_copy.imag = fft_bringe_imag
        real_image = fftpack.ifft(fft_copy)

        image_res = Image.fromarray(real_image.astype('uint8'), mode='L')

        if (os.path.exists(r"vosst.tif")):
            os.remove(r"vosst.tif")

        image_res.save(r"vosst.tif")

        image_res.close()
        again_bridge_image = io.imread(r"vosst.tif")
        br_np = np.asarray(again_bridge_image)
        fft_image2 = fftpack.fft(br_np)
        f_tilda_imag=fft_image2.imag

        # vect_fw = np.array([])
        # vect_f = np.array([])
        #
        # for i in range(0, 128):
        #     vect_fw = np.append(vect_fw, [fft_image2[128 + i, 256 - i:257 + i]])
        #     vect_f = np.append(vect_f, [fft_bridge[128 + i, 256 - i:257 + i]])
        #
        # for i in range(0, 128):
        #     vect_fw = np.append(vect_fw, [fft_image2[384 - i, 256 - i:257 + i]])
        #     vect_f = np.append(vect_f, [fft_bridge[384 - i, 256 - i:257 + i]])

        # vect_f = fftpack.fft(vect_f)
        # vect_fw = fftpack.fft(vect_fw)

        f_signal_tilda=f_tilda_imag[h_l_board:h_r_board, w_l_board:w_r_board]

        f_W_tilda_vec=np.ravel(f_signal_tilda, order='C')

        f_container=fft_bridge[h_l_board:h_r_board, w_l_board:w_r_board]

        f_container_vec=np.ravel(f_container)

        # vect_f=vect_f.imag
        # vect_fw=vect_fw.imag

        f_container_vec_image=f_container_vec.imag
        #omega_tilda = OmegaTilda(vect_fw, vect_f , Alfa)
        omega_tilda=OmegaTilda(f_W_tilda_vec,f_container_vec_image)

        omega_tilda_imag=omega_tilda.imag

        ro=test_ro(omega,omega_tilda)

        print(f"alfa={Alfa} , ro={ro} ,psnr={psnr(bridge_np, real_image.real)}")


        if(ro>0.4 and psnr(bridge_np,real_image.real)>30):
            cprint('DAAAAAAAAAAAA!', 'green', 'on_red')
            print(f"alfa={Alfa} , ro={ro}")













