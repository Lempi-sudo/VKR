import numpy as np
from skimage.io import imshow, show
from matplotlib import pyplot as plt
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter



def checkboard_maker(w, h, cell, f, s):
    board = [[f, s] * int(w / cell / 2), [s, f] * int(w / cell / 2)] * int(h / cell / 2)
    board = np.kron(board, np.ones((cell, cell)))
    return board

def SKO(bb, perfb):
    median_sko = ((np.sum(bb) - np.sum(perfb)) ** 2) / (M * N)
    return  median_sko

def СoefficientDecreaseNoise(fb,perfb,badb):
    res=np.mean((fb - perfb) ** 2) / np.mean((badb - perfb) ** 2)
    return res


def filterAddNoise(d2):
    # генерируем доску
    print()
    print()
    if d2 == 1:
        f = 160/255
        s = 96/255
    else:
        f = 0.75
        s = 0.31

    board = checkboard_maker(M, N, cell, f, s)

    # дисперсия доски
    dispersion = np.nanvar(board)
    print("Дисперсия исх изображения", dispersion)
    d=np.sqrt(dispersion)

    # портим изображение аддитивный белый шум (Гаус)
    bad_board = random_noise(board, var=dispersion / d2)
    noise = board - bad_board

    noise_dis=np.nanvar(noise)
    print("Дисперсия  шума", noise_dis)

    n=np.sqrt(noise_dis)

    # медианный фильтр
    median_board = median_filter(bad_board, size=3)
    median_sko = ((np.sum(median_board) - np.sum(board)) ** 2) / (M * N)
    print("(Медианный) Квадрат СКО: ", median_sko)
    median_ksh = np.mean((median_board - board) ** 2) / np.mean((bad_board - board) ** 2)
    print("(Медианный) Коеффициент снижения шума: ", median_ksh)


    # линейный сглаживающий фильтр
    mask = np.array([[0.06, 0.1, 0.06], [0.1, 0.36, 0.1], [0.06, 0.1, 0.06]])
    linear_board = convolve2d(bad_board, mask, boundary='symm', mode='same')
    linear_sko = ((np.sum(linear_board) - np.sum(board)) ** 2) / (M * N)
    print("(Линейный) Квадрат СКО: ", linear_sko)
    median_ksh = np.mean((median_board - board) ** 2) / np.mean((bad_board - board) ** 2)
    print("(Линейный) Коеффициент снижения шума: ", median_ksh)


    fig = plt.figure(figsize=(8, 9))
    fig.add_subplot(3, 2, 1)
    plt.title('Исходное шахматное поле')
    imshow(board, cmap='gray')
    fig.add_subplot(3, 2, 2)
    plt.title('Шахматное поле зашумленное')
    imshow(bad_board, cmap='gray')
    fig.add_subplot(3, 2, 3)
    plt.title('Зашумленное после медианного фильтра')
    imshow(median_board, cmap='gray')
    fig.add_subplot(3, 2, 4)
    plt.title('Зашумленное после линейного фильтра')
    imshow(linear_board, cmap='gray')
    fig.add_subplot(3, 2, 5)
    plt.title('Шумы')
    imshow(noise, cmap='gray')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.4)
    show()


def filterImpulse(p):
    print()
    print()
    # генерируем доску
    f = 0
    s = 1
    board = checkboard_maker(M, N, cell, f, s)

    dispersion = np.nanvar(board)
    print("Дисперсия исх изображения", dispersion)

    noise = random_noise(np.full(board.shape, -1), mode='s&p', amount=p)

    bad_board = board.copy()

    for index,x in np.ndenumerate(board):
        if noise[index] == -1:
            bad_board[index]=0
        if noise[index] == 1:
            bad_board[index] = 1

    noise_dis = np.nanvar(noise)
    print("Дисперсия  шума", noise_dis)

    # медианный фильтр
    median_board = median_filter(bad_board, size=3)
    median_sko = ((np.sum(median_board) - np.sum(board)) ** 2) / (M * N)
    print("(Медианный) Квадрат СКО: ", median_sko)
    median_ksh = np.mean((median_board - board) ** 2) / np.mean((bad_board - board) ** 2)
    print("(Медианный) Коеффициент снижения шума: ", median_ksh)

    # линейный сглаживающий фильтр
    mask = np.array([[0.06, 0.1, 0.06], [0.1, 0.36, 0.1], [0.06, 0.1, 0.06]])
    linear_board = convolve2d(bad_board, mask, boundary='symm', mode='same')
    linear_sko = ((np.sum(linear_board) - np.sum(board)) ** 2) / (M * N)
    print("(Линейный) Квадрат СКО: ", linear_sko)
    median_ksh = np.mean((linear_board - board) ** 2) / np.mean((bad_board - board) ** 2)
    print("(Медианный) Коеффициент снижения шума: ", median_ksh)


    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(3, 2, 1)
    plt.title('Исходное шахматное поле')
    imshow(board, cmap='gray')
    fig.add_subplot(3, 2, 2)
    plt.title('Шахматное поле зашумленное')
    imshow(bad_board, cmap='gray')
    fig.add_subplot(3, 2, 3)
    plt.title('Зашумленное после медианного фильтра')
    imshow(median_board, cmap='gray')
    fig.add_subplot(3, 2, 4)
    plt.title('Зашумленное после линейного фильтра')
    imshow(linear_board, cmap='gray')
    fig.add_subplot(3, 2, 5)
    plt.title('Шум')
    imshow(noise, cmap='gray')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    show()



M = 128
N = 128
cell = 16


filterAddNoise(10)
filterAddNoise(1)
filterImpulse(0.1)
filterImpulse(0.3)
