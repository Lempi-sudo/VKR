import numpy as np
import cv2

def MSE_image(x,y):
    e= (np.sum((x - y) ** 2)) / (len(x) * len(y[0]))
    return e

def psnr(W, Wr):
 e = (np.sum((W - Wr) ** 2)) / (len(W) * len(W[0]))
 p = 10 * np.log10(255 ** 2 / e)
 return p


def cv2PSRN(W, Wr):
     p=cv2.PSNR(W,Wr)
     return p

#результат выдает побитовое сравнение в процентах насколько похожи два ЦВЗ
def pobitovo_sravnenie_WaterMark(W1, W2 , total_bit=1024):
    t1 = W1==W2
    sum=t1.sum()
    return sum/total_bit*100






