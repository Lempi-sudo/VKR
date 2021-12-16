from scipy.ndimage import median_filter
import numpy as np
from matplotlib import pyplot as plt
import os
from termcolor import cprint
from skimage import io
from PIL import Image
from scipy import fftpack
from prettytable import PrettyTable
import cv2




def test_true_ro():
    res_ro=0
    alfa_res=0
    omega = generationOmega(512, 1 / 4, 1 / 2)
    omega_matrica = omega.reshape(128, 256)
    AlfaList = np.arange(5.0, 32.0, 0.1, dtype=float)


    for Alfa in AlfaList:
        bridge_image = Image.open(r"barb.tif")
        bridge_np = np.asarray(bridge_image)
        #fft_bridge = fftpack.fft(bridge_np)
        fft_bridge = np.fft.fft2(bridge_np)

        container_copy=fft_bridge.copy()
        fft_copy = fft_bridge.copy()



        fft_copy_real = insert_siganl_into_container(container_copy.real,omega_matrica,Alfa)

        fft_copy.real=fft_copy_real
        real_image_ifft=np.fft.ifft2(fft_copy)

        real_image = real_image_ifft.real

        image_res = Image.fromarray(real_image.astype('uint8'), mode='L')

        if (os.path.exists(r"CW.tif")):
            os.remove(r"CW.tif")
        image_res.save(r"CW.tif")
        image_res.close()

        image_out = io.imread(r"CW.tif")

        CW_tilda_np = np.asarray(image_out)
        fft_CW_Tilda = np.fft.fft2(CW_tilda_np)
        fft_CW_Tilda_real = fft_CW_Tilda.real



        f_signal_tilda = extract_signal(fft_CW_Tilda_real,omega_matrica)
        f_container = extract_signal(fft_bridge, omega_matrica)


        f_container_vec_real = f_container.real

        omega_tilda = OmegaTilda(f_signal_tilda, f_container_vec_real, Alfa)

        ro = test_ro(omega_matrica, omega_tilda)

        if (ro > 0.5 and psnr(bridge_np, real_image.real) > 28):
            cprint('DAAAAAAAAAAAA!', 'green', 'on_red')
            print(f"alfa={Alfa} , ro={ro}  ,psnr={psnr(bridge_np, real_image.real)}")

        if ro > res_ro and psnr(bridge_np, real_image.real)>28 :
            res_ro=ro
            alfa_res=Alfa

    return  res_ro , omega , alfa_res


# def insert_signal_into_container(container :np.ndarray, omega_matrica : np.ndarray , Alfa):
#     Hight_image = container.shape[0]
#     Witht_image = container.shape[1]
#     Witht_signal = omega_matrica.shape[0]
#     Hight_signal = omega_matrica.shape[1]
#     h_l_board = (Hight_image // 2) - (Hight_signal // 2)
#     h_r_board = (Hight_image // 2) + (Hight_signal // 2)
#     w_l_board = (Witht_image // 2) - (Witht_signal // 2)
#     w_r_board = (Witht_image // 2) + (Witht_signal // 2)
#
#     result=container.copy()
#
#     result[w_l_board:w_r_board, h_l_board:h_r_board] = container[w_l_board:w_r_board,h_l_board:h_r_board] * (1 + Alfa * omega_matrica)
#
#     return  result

# def extract_signal(container :np.ndarray ,  omega_matrica : np.ndarray):
#     Hight_image = container.shape[0]
#     Witht_image = container.shape[1]
#     Witht_signal = omega_matrica.shape[0]
#     Hight_signal = omega_matrica.shape[1]
#     h_l_board = (Hight_image // 2) - (Hight_signal // 2)
#     h_r_board = (Hight_image // 2) + (Hight_signal // 2)
#     w_l_board = (Witht_image // 2) - (Witht_signal // 2)
#     w_r_board = (Witht_image // 2) + (Witht_signal // 2)
#
#     return  container[w_l_board:w_r_board, h_l_board:h_r_board]


def extract_signal(container :np.ndarray ,  omega_matrica : np.ndarray):

    vect=np.array([])

    for i in range(0, 128):
        vect = np.append(vect, [container[128 + i, 256 - i:257 + i]])

    for i in range(0, 128):
        vect = np.append(vect, [container[384 - i, 256 - i:257 + i]])


    return  vect.reshape(128,256)


def insert_siganl_into_container(container :np.ndarray, omega_matrica : np.ndarray , Alfa):

    omega=omega_matrica.ravel()

    container_res=container.copy()

    container_res=container_res.real

    sum = 0
    for i in range(0, 128):
        sum += (len(container_res[128 + i, 256 - i:257 + i]))

        container_res[128 + i, 256 - i:257 + i] = container.real[128 + i, 256 - i:257 + i] * (
                    1 + Alfa * omega[i * i:(i + 1) * (i + 1)])

    for i in range(0, 128):
        sum += (len(container_res[384 - i, 256 - i:257 + i]))
        container_res[384 - i, 256 - i:257 + i] = container.real[384 - i, 256 - i:257 + i] * (
                    1 + Alfa * omega[128 * 128 + i * i: 128 * 128 + (i + 1) * (i + 1)])

    return container_res


def psnr(W, Wr):
 e = (np.sum((W - Wr) ** 2)) / (len(W) * len(W[0]))
 p = 10 * np.log10(255 ** 2 / e)
 return p
def OmegaTilda(fW_tilda ,f ,a=1):
    return np.where(f!=0 ,((fW_tilda-f) / (a*f)),255)
def test_ro(Q , QTilda):
    t1=np.sum(Q*QTilda)
    t2=np.sqrt(np.sum(Q*Q))
    t3=np.sqrt(np.sum(QTilda*QTilda))
    t4=t3*t2
    res =t1/t4
    return  res
def generationOmega(size,C,H , loc=0.0 ,scale = 1.0 ):
    size_cvz = size * size
    CH = C * H
    size_omega = size_cvz * CH
    omega = np.random.normal(loc=loc, scale=scale,size=int(size_omega))
    return omega
def draw(x,y , title = ""):
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel('Y = значения ro')
    plt.xlabel('X параметр искажения ')
    plt.show()

def Task4_saveJpeg(omega , alfa , ro):
    Y = []
    X = []
    for QF in range(30, 110, 10):
        omega_matrica = omega.reshape(128, 256)
        Alfa = alfa

        bridge_image = Image.open(r"barb.tif")
        bridge_np = np.asarray(bridge_image)
        # fft_bridge = fftpack.fft(bridge_np)
        fft_bridge = np.fft.fft2(bridge_np)

        fft_copy = fft_bridge.copy()
        fft_bringe_real = fft_copy.real

        Hight_image = 512
        Witht_image = 512
        Witht_signal = 128
        Hight_signal = 256
        h_l_board = (Hight_image // 2) - (Hight_signal // 2)
        h_r_board = (Hight_image // 2) + (Hight_signal // 2)
        w_l_board = (Witht_image // 2) - (Witht_signal // 2)
        w_r_board = (Witht_image // 2) + (Witht_signal // 2)

        f = fft_bringe_real[w_l_board:w_r_board, h_l_board:h_r_board]

        fft_bringe_real[w_l_board:w_r_board, h_l_board:h_r_board] = fft_bringe_real[w_l_board:w_r_board,
                                                                    h_l_board:h_r_board] * (1 + Alfa * omega_matrica)

        fft_copy.real = fft_bringe_real

        real_image_ifft = np.fft.ifft2(fft_copy)

        real_image = real_image_ifft.real

        image_res = Image.fromarray(real_image.astype('uint8'), mode='L')

        if (os.path.exists(r"savejpeg.jpeg")):
            os.remove(r"savejpeg.jpeg")

        image_res.save("savejpeg.jpeg", quality=QF)

        image_res.close()

        CW_tilda = io.imread(r"savejpeg.jpeg")

        CW_tilda_np = np.asarray(CW_tilda)
        fft_CW_Tilda = np.fft.fft2(CW_tilda_np)
        fft_CW_Tilda_real = fft_CW_Tilda.real

        f_signal_tilda = fft_CW_Tilda_real[w_l_board:w_r_board, h_l_board:h_r_board]

        f_container = fft_bridge[w_l_board:w_r_board, h_l_board:h_r_board]

        f_container_vec_real = f_container.real

        omega_tilda = OmegaTilda(f_signal_tilda, f_container_vec_real, Alfa)

        ro = test_ro(omega_matrica, omega_tilda)

        X.append(QF)

        Y.append(ro)

    draw(X, Y, title="Формат jpeg")

def Task3_Median(omega , alfa ,ro):
    Y = []
    X = []
    omega_matrica = omega.reshape(128, 256)
    Alfa=alfa
    for M in range(1,15,2):
        bridge_image = Image.open(r"barb.tif")
        bridge_np = np.asarray(bridge_image)
        fft_bridge = np.fft.fft2(bridge_np)

        fft_copy = fft_bridge.copy()
        fft_bringe_real = fft_copy.real

        Hight_image = 512
        Witht_image = 512
        Witht_signal = 128
        Hight_signal = 256
        h_l_board = (Hight_image // 2) - (Hight_signal // 2)
        h_r_board = (Hight_image // 2) + (Hight_signal // 2)
        w_l_board = (Witht_image // 2) - (Witht_signal // 2)
        w_r_board = (Witht_image // 2) + (Witht_signal // 2)

        fft_bringe_real[w_l_board:w_r_board, h_l_board:h_r_board] = fft_bringe_real[w_l_board:w_r_board,h_l_board:h_r_board] * (1 + Alfa * omega_matrica)

        fft_copy.real = fft_bringe_real

        real_image_ifft = np.fft.ifft2(fft_copy)

        real_image = real_image_ifft.real

        image_res = Image.fromarray(real_image.astype('uint8'), mode='L')

        if (os.path.exists(r"CW.tif")):
            os.remove(r"CW.tif")
        image_res.save(r"CW.tif")
        image_res.close()

        image_out = io.imread(r"CW.tif")

        CW_tilda_np = np.asarray(image_out)

        CW_tilda_median = median_filter(CW_tilda_np, size=int(M))

        #CW_tilda_median=CW_tilda_np  # если уберу медианное искажение ro "хорошим"

        image_res_median = Image.fromarray(CW_tilda_median.astype('uint8'), mode='L')
        image_res_median.save(r"median.tif")

        fft_CW_Tilda = np.fft.fft2(CW_tilda_median)
        fft_CW_Tilda_real = fft_CW_Tilda.real

        f_signal_tilda = fft_CW_Tilda_real[w_l_board:w_r_board, h_l_board:h_r_board]

        f_container = fft_bridge[w_l_board:w_r_board, h_l_board:h_r_board]

        f_container_vec_real = f_container.real

        omega_tilda = OmegaTilda(f_signal_tilda, f_container_vec_real, Alfa)

        ro = test_ro(omega_matrica, omega_tilda)


        X.append(M)

        Y.append(ro)

    draw(X, Y, title="медианное искажение")

def cutImage(CW: np.ndarray , C: np.ndarray  , V: float ):

    cwcopy=CW.copy()
    n1=int(CW.shape[0] * np.sqrt(V))
    n2=int(CW.shape[1] *np.sqrt(V))
    cwcopy[0:n1,0:n2]=CW[0:n1,0:n2]
    cwcopy[n1:CW.shape[0] , n2:CW.shape[1] ] = C[n1:CW.shape[0] , n2:CW.shape[1]]
    return  cwcopy
def Task1_Cut(omega , alfa , ro ):
    Y = []
    X = []
    VlIST=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]
    omega_matrica = omega.reshape(128,256)
    Alfa = alfa
    for V in VlIST:
        bridge_image = Image.open(r"barb.tif")
        bridge_np = np.asarray(bridge_image)
        # fft_bridge = fftpack.fft(bridge_np)
        fft_bridge = np.fft.fft2(bridge_np)

        fft_copy = fft_bridge.copy()
        fft_bringe_real = fft_copy.real

        Hight_image = 512
        Witht_image = 512
        Witht_signal = 128
        Hight_signal = 256
        h_l_board = (Hight_image // 2) - (Hight_signal // 2)
        h_r_board = (Hight_image // 2) + (Hight_signal // 2)
        w_l_board = (Witht_image // 2) - (Witht_signal // 2)
        w_r_board = (Witht_image // 2) + (Witht_signal // 2)

        f = fft_bringe_real[w_l_board:w_r_board, h_l_board:h_r_board]

        fft_bringe_real[w_l_board:w_r_board, h_l_board:h_r_board] = fft_bringe_real[w_l_board:w_r_board,
                                                                    h_l_board:h_r_board] * (1 + Alfa * omega_matrica)

        fft_copy.real = fft_bringe_real

        real_image_ifft = np.fft.ifft2(fft_copy)

        real_image = real_image_ifft.real

        image_res = Image.fromarray(real_image.astype('uint8'), mode='L')

        if (os.path.exists(r"CW.tif")):
            os.remove(r"CW.tif")
        image_res.save(r"CW.tif")
        image_res.close()

        image_out = io.imread(r"CW.tif")

        image_out=np.asarray(image_out)

        image_cut = cutImage(image_out, bridge_np, V)

        CW_tilda_np = image_cut
        fft_CW_Tilda = np.fft.fft2(CW_tilda_np)
        fft_CW_Tilda_real = fft_CW_Tilda.real

        f_signal_tilda = fft_CW_Tilda_real[w_l_board:w_r_board, h_l_board:h_r_board]

        f_container = fft_bridge[w_l_board:w_r_board, h_l_board:h_r_board]

        f_container_vec_real = f_container.real

        omega_tilda = OmegaTilda(f_signal_tilda, f_container_vec_real, Alfa)

        ro = test_ro(omega_matrica, omega_tilda)
        X.append(V)

        Y.append(ro)

    draw(X, Y, title="CUT")
def Scale(omega , alfa , ro ):
    Y = []
    X = []
    omega_matrica = omega.reshape(128, 256)
    Alfa = alfa
    p_list = np.arange(0.55, 1.45, 0.15)

    bridge_image = Image.open(r"barb.tif")
    bridge_np = np.asarray(bridge_image)
    # fft_bridge = fftpack.fft(bridge_np)
    fft_bridge = np.fft.fft2(bridge_np)

    container_copy = fft_bridge.copy()
    fft_copy = fft_bridge.copy()

    fft_copy_real = insert_siganl_into_container(container_copy.real, omega_matrica, Alfa)

    fft_copy.real = fft_copy_real
    real_image_ifft = np.fft.ifft2(fft_copy)

    real_image = real_image_ifft.real

    image_res = Image.fromarray(real_image.astype('uint8'), mode='L')

    if (os.path.exists(r"CW.tif")):
        os.remove(r"CW.tif")
    image_res.save(r"CW.tif")
    image_res.close()

    image_out = io.imread(r"CW.tif")

    CW_tilda_np = np.asarray(image_out)

    N = np.shape(CW_tilda_np)[0]
    CW_zeros = np.zeros((N, N))

    for p in p_list:
        CWscale=CW_tilda_np.copy()
        new_size = int(N * p)
        if p <= 1:
            CWscale[new_size:N,:] = 0.0
            CWscale[:,new_size:N] = 0.0
        else:
            CWscale_big = cv2.resize(CWscale, (new_size, new_size))
            CWscale = CWscale_big[0:N , 0:N]

        fft_CW_Tilda = np.fft.fft2(CWscale)
        fft_CW_Tilda_real = fft_CW_Tilda.real

        f_signal_tilda = extract_signal(fft_CW_Tilda_real, omega_matrica)
        f_container = extract_signal(fft_bridge, omega_matrica)

        f_container_vec_real = f_container.real

        omega_tilda = OmegaTilda(f_signal_tilda, f_container_vec_real, Alfa)

        ro = test_ro(omega_matrica, omega_tilda)

        X.append(p)

        Y.append(ro)


    draw(X, Y, title="Масштабирование")
def Two_distortion(omega , alfa , ro ):
    name=['cut/scale','0.55','0.70','0.85','1.0','1.15','1.3','1.45']
    mytable = PrettyTable(name)

    p_list=[0.55,0.70,0.85,1.0,1.15,1.3 , 1.45]
    VlIST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    omega_matrica = omega.reshape(128, 256)

    Alfa = alfa
    for V in VlIST:
        X = []
        bridge_image = Image.open(r"barb.tif")
        bridge_np = np.asarray(bridge_image)
        # fft_bridge = fftpack.fft(bridge_np)
        fft_bridge = np.fft.fft2(bridge_np)

        fft_copy = fft_bridge.copy()
        fft_bringe_real = fft_copy.real

        Hight_image = 512
        Witht_image = 512
        Witht_signal = 128
        Hight_signal = 256
        h_l_board = (Hight_image // 2) - (Hight_signal // 2)
        h_r_board = (Hight_image // 2) + (Hight_signal // 2)
        w_l_board = (Witht_image // 2) - (Witht_signal // 2)
        w_r_board = (Witht_image // 2) + (Witht_signal // 2)

        f = fft_bringe_real[w_l_board:w_r_board, h_l_board:h_r_board]

        fft_bringe_real[w_l_board:w_r_board, h_l_board:h_r_board] = fft_bringe_real[w_l_board:w_r_board,
                                                                    h_l_board:h_r_board] * (1 + Alfa * omega_matrica)

        fft_copy.real = fft_bringe_real

        real_image_ifft = np.fft.ifft2(fft_copy)

        real_image = real_image_ifft.real

        image_res = Image.fromarray(real_image.astype('uint8'), mode='L')

        if (os.path.exists(r"CW.tif")):
            os.remove(r"CW.tif")
        image_res.save(r"CW.tif")
        image_res.close()

        image_out = io.imread(r"CW.tif")

        image_out = np.asarray(image_out)

        image_cut = cutImage(image_out, bridge_np, V)

        N = np.shape(image_cut)[0]

        X.append(V)
        for p in p_list:
            CWscale = image_cut.copy()
            new_size = int(N * p)
            if p <= 1:
                CWscale[new_size:N, :] = 0.0
                CWscale[:, new_size:N] = 0.0
            else:
                CWscale_big = cv2.resize(CWscale, (new_size, new_size))
                CWscale = CWscale_big[0:N, 0:N]

            fft_CW_Tilda = np.fft.fft2(CWscale)
            fft_CW_Tilda_real = fft_CW_Tilda.real



            f_signal_tilda = fft_CW_Tilda_real[w_l_board:w_r_board, h_l_board:h_r_board]

            f_container = fft_bridge[w_l_board:w_r_board, h_l_board:h_r_board]

            f_container_vec_real = f_container.real

            omega_tilda = OmegaTilda(f_signal_tilda, f_container_vec_real, Alfa)

            ro = test_ro(omega_matrica, omega_tilda)
            X.append(ro)

        mytable.add_row(X[:])

    print(mytable)










if __name__ == '__main__':
    res_ro , omega , alfa_res =test_true_ro()

    print(f"max ro ={res_ro}")
    print(f"Alfa = {alfa_res}")

    Task1_Cut(omega,alfa_res,res_ro)
    Scale(omega, alfa_res, res_ro)
    Task3_Median(omega, alfa_res, res_ro)
    Task4_saveJpeg(omega, alfa_res ,res_ro)
    Two_distortion(omega,alfa_res,res_ro)


