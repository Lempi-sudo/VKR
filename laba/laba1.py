import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from PIL import Image, ImageDraw
from skimage.util import random_noise
from skimage.exposure import histogram, equalize_hist
from skimage import color

def rgb2ycbcr222(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
def rgb2ycbcrtest(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
def MyYCBCRTORGB(im):
    res = im.copy()
    Y = im[:, :, 0]
    Cb = im[:, :, 1]
    Cr = im[:, :, 2]

    R = Y + 1.402*Cr

    B = Y + 1.772*Cb

    G = Y - 0.344*Cb - 0.714*Cr



    res[:, :, 0] = R
    res[:, :, 1] = G
    res[:, :, 2] = B

    return res
def Myrgb2TOycbcr(im):
    res=im.copy()
    r=im[:,:,0]
    g = im[:, :, 1]
    b = im[:, :, 2]

    Y=(77*r)/256 + (150*g)/256 + (29*b)/256

    cb=b-Y

    cr=r--Y

    res[:,:,0]=Y
    res[:, :, 1] = cb
    res[:, :, 2] = cr

    return res
def convert(img): #ПРОВЕРКА ФУНКЦИЙ КОНВЕРТАЦИИ  color.rgb2ycbcr  И color.ycbcr2rgb
    image_ycrcrb = color.rgb2ycbcr(img)
    img1 = Image.fromarray(image_ycrcrb.astype(np.uint8))
    img1.show(title="пустой контейнер")

    img2=color.ycbcr2rgb(image_ycrcrb)
    img2*=255
    img2 = Image.fromarray(img2.astype(np.uint8))
    img2.show(title="пустой контейнер")

def svi_4_v_2():
    delta=4

    ornament=Image.open(r"Image\ornament.tif")
    secret_ornament = np.mod(ornament, 2)

    Wnn=secret_ornament


    image = Image.open(r"Image\baboon.tif")
    Image_RGB = image.convert('RGB')
    Image_RGB_np = np.asarray(Image_RGB)

    image_ycrcrb = Myrgb2TOycbcr(Image_RGB_np)
    img = Image.fromarray(image_ycrcrb.astype(np.uint8))
    img.show(title="пустой контейнер")

    Image_Copy=image_ycrcrb.copy()

    Cr_channel= image_ycrcrb[:,:,2]

    Cnn=Cr_channel

    Vnn=np.mod(Cnn,delta)

    CW=(Cnn//(2*delta))*(2*delta)+(Wnn*delta)+Vnn


    img = Image.fromarray(CW.astype(np.uint8))
    img.show(title="встроенное цвз по одному каналу")

    image_res=Image_Copy.copy()
    image_res[:,:,2]=CW

    img = Image.fromarray(image_res.astype(np.uint8))
    img.show(title="итоговый контейнер со встроенным ЦВЗ")

    # ИЗВЛЕКАЕМ ЦВЗ 2
    W = CW - (Cnn // (2 * delta))*(2*delta) - Vnn

    W=W//delta

    W[W==1]=255

    img = Image.fromarray(W.astype(np.uint8))
    img.show(title="извлеченный ЦВЗ")
def svi_4_v_3():
    delta=4

    ornament=Image.open(r"Image\ornament.tif")
    secret_ornament = np.mod(ornament, 2)
    secret_ornament[secret_ornament>=1]=255

    Wnn=secret_ornament


    image = Image.open(r"Image\baboon.tif")
    Image_RGB = image.convert('RGB')
    Image_RGB_np = np.asarray(Image_RGB)

    image_ycrcrb=color.rgb2ycbcr(Image_RGB_np)
    Image_Copy=image_ycrcrb.copy()

    Cr_channel= image_ycrcrb[:,:,2]
    Cnn=Cr_channel

    Vnn=np.mod(Cnn,delta)

    CW=(Cnn//(2*delta))*(2*delta)+(Wnn*delta)+Vnn

    img = Image.fromarray(image_ycrcrb.astype(np.uint8))
    img.show()

    img = Image.fromarray(CW.astype(np.uint8))
    img.show()

    image_res=Image_Copy.copy()
    image_res[:,:,2]=CW

    imagergb=color.ycbcr2rgb(image_res)
    imagergb=imagergb*255
    img = Image.fromarray(imagergb.astype(np.uint8))

    img.show()

    img = Image.fromarray(image_res.astype(np.uint8))
    img.show()

    # ИЗВЛЕКАЕМ ЦВЗ 2
    W = CW - (Cnn // (2 * delta))*(2*delta) - Vnn
    W=W//delta


    W[W>=1]=255

    img = Image.fromarray(W.astype(np.uint8))
    img.show()
def svi_4():
    delta=4

    ornament=Image.open(r"ornament.tif")
    secret_ornament = np.mod(ornament, 2)

    Wnn=secret_ornament


    image = Image.open(r"baboon.tif")
    Image_RGB = image.convert('RGB')
    Image_RGB_np = np.asarray(Image_RGB)

    image_ycrcrb = color.rgb2ycbcr(Image_RGB_np)
    img = Image.fromarray(image_ycrcrb.astype(np.uint8))
    img.show(title="пустой контейнер")

    Image_Copy=image_ycrcrb.copy()

    Cr_channel= image_ycrcrb[:,:,2]

    Cnn=Cr_channel

    Vnn=np.mod(Cnn,delta)

    CW=(Cnn//(2*delta))*(2*delta)+(Wnn*delta)+Vnn


    img = Image.fromarray(CW.astype(np.uint8))
    img.show(title="встроенное цвз по одному каналу")

    image_res=Image_Copy.copy()
    image_res[:,:,2]=CW

    img = Image.fromarray(image_res.astype(np.uint8))
    img.show(title="итоговый контейнер со встроенным ЦВЗ")

    img1=color.ycbcr2rgb(image_res)
    img1=img1*255
    img = Image.fromarray(img1.astype(np.uint8))
    img.show(title="итоговый контейнер со встроенным ЦВЗ")


    # ИЗВЛЕКАЕМ ЦВЗ 2
    W = CW - (Cnn // (2 * delta))*(2*delta) - Vnn

    W=W//delta

    W[W==1]=255

    img = Image.fromarray(W.astype(np.uint8))
    img.show(title="извлеченный ЦВЗ")
def svi_1():
    #считываем контейнер
    image = Image.open(r"baboon.tif")
    Image_RGB = image.convert('RGB')
    Image_RGB_np = np.asarray(Image_RGB)
    #исходное изображение копия
    CopyImage = Image_RGB_np.copy()

    #считывание ЦВЗ орнамент
    secret_red_Image=Image.open(r"ornament.tif")
    secret_ornament=np.mod(secret_red_Image,2)

    #считывание ЦВЗ микки
    secret_green_Image=Image.open(r"mickey.tif")
    secret_mickey=np.mod(secret_green_Image,2)

    #канал green
    green= Image_RGB_np[:,:,1]
    #по последнему биту формируем матрицу (0,1)
    green_nzb=np.mod(green, 2)


    #ксорим с секретной инфой
    xor_green_nzb = np.logical_xor(green_nzb, secret_mickey)
    xor_green_nzb = np.array(xor_green_nzb, dtype=int)

    #зануляем последий бит в контейнере
    green=(green//2)*2

    # складываем секретную инфу с цветовым каналом Green
    CWGreen = green + xor_green_nzb
    # получаем цветовой канал  Green c ЦВЗ

    #ОТОБРАЖАЕМ ИЗОБРАЖЕНИЕ ПО ЗЕЛЕНОМУ КАНАЛУ
    Image_Green_with_nzb=CopyImage.copy()
    Image_Green_with_nzb[:,:,1]=CWGreen
    Image_Green_with_nzb[:, :, 0]=0
    Image_Green_with_nzb[:, :, 2]=0
    imggreen = Image.fromarray(Image_Green_with_nzb.astype(np.uint8))
    imggreen.show()



    #канал red
    red= Image_RGB_np[:,:,0]
    #по последнему биту формируем матрицу (0,1)
    red_nzb=np.mod(red, 2)

    #ксорим с секретной инфой
    xor_red_nzb = np.logical_xor(red_nzb, secret_ornament)
    xor_red_nzb = np.array(xor_red_nzb, dtype=int)

    #зануляем последий бит в контейнере
    red=(red//2)*2

    #складываем секретную инфу с цветовым каналом red
    CWred=red+xor_red_nzb
    # получаем цветовой канал  red c ЦВЗ

    #ОТОБРАЖАЕМ ИЗОБРАЖЕНИЕ ПО КРАСНОМУ  КАНАЛУ
    Image_RED_with_nzb=CopyImage.copy()
    Image_RED_with_nzb[:,:,0]=CWred
    Image_RED_with_nzb[:, :, 1]=0
    Image_RED_with_nzb[:, :, 2]=0
    imgRed = Image.fromarray(Image_RED_with_nzb.astype(np.uint8))
    imgRed.show()

    #отображаем изображение с двумя вставленными цвз

    Image_full=CopyImage.copy()
    Image_full[:,:,0]=CWred
    Image_full[:,:,1]=CWGreen
    imgfull = Image.fromarray(Image_full.astype(np.uint8))
    imgfull.show()



    #извлекаем информацию орнамент
    cw=Image_full[:,:,0]
    c=CopyImage[:,:,0]

    cw=np.mod(cw,2)
    c = np.mod(c, 2)

    w=np.logical_xor(cw,c)

    w=np.array(w, dtype=int)
    w[w>=1]=255
    image_res=Image.fromarray(w.astype('uint8'))
    image_res.show()

    #извлекаем информацию МИККИ
    cw=Image_full[:,:,1]
    c=CopyImage[:,:,1]

    cw=np.mod(cw,2)
    c = np.mod(c, 2)

    w=np.logical_xor(cw,c)

    w=np.array(w, dtype=int)
    w[w>=1]=255
    image_res=Image.fromarray(w.astype('uint8'), mode='L')
    image_res.show()




if __name__ == '__main__':
    svi_4()
    svi_1()
