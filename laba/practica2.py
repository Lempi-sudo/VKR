import numpy as np
from skimage.io import imread, imshow, show, imsave
from scipy.fftpack import dctn, idctn
from scipy import signal
from matplotlib import pyplot as plt
import random
import cv2

bridge = imread('bridge.tif')
ornament = imread('ornament.tif')
logo = imread('logo.bmp')
logo[logo > 0] = 1

C = np.array(np.copy(bridge), dtype=float)
CW = np.array(np.copy(ornament), dtype=float)


def psnr(W, Wr):
 e = (np.sum((W - Wr) ** 2)) / (len(W) * len(W[0]))
 p = 10 * np.log10(255 ** 2 / e)
 return p



def simple_dct_embed(C, logo):
  h_logo = logo.shape[0]
  w_logo = logo.shape[1]
  h_C = C.shape[0]
  w_C = C.shape[1]
  CDOT = dctn(C, norm='ortho')
  # logo_DOT = dctn(logo, norm='ortho')

  h_l_board = (h_C // 2) - (h_logo // 2)
  h_r_board = (h_C // 2) + (h_logo // 2)
  w_l_board = (w_C // 2) - (w_logo // 2)
  w_r_board = (w_C // 2) + (w_logo // 2)
  CDOT[h_l_board:h_r_board, w_l_board:w_r_board] = logo
  Cw = idctn(CDOT, norm='ortho')
  plt.imshow(Cw, cmap='gray')
  plt.show()
  return Cw


value = psnr(C, CW)
print(value)
print(cv2.PSNR(C, CW))

Cw = simple_dct_embed(C, logo)


plt.imsave("cw.png", Cw)
Cw_png = imread('cw.png').astype(float)
Cw_png = Cw_png[:, :, 1]
error = np.abs(C - Cw_png)
error_max = np.max(error)
plt.imshow(error, cmap='gray')
plt.show()
print(cv2.PSNR(C, Cw_png))

CwDOT = dctn(Cw, norm='ortho')
CwDOT[CwDOT > 1] = 1
CwDOT[CwDOT < 0] = 0
plt.imshow(CwDOT, cmap='gray')
plt.show()
