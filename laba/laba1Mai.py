import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage.exposure import histogram, equalize_hist


def a():
    path = '12_tank.tif'
    image = io.imread(path)
    fig = plt.figure(figsize=(10, 5))
    ax_1 = fig.add_subplot(3, 1, 1)
    ax_1.imshow(image,cmap= 'gray')
    ax_2 = fig.add_subplot(3,1, 2)
    tank=image.T
    ax_2.imshow(tank,cmap='gray')
    ax_3=fig.add_subplot(3,1,3)
    ax_3.imshow(image,cmap='gray')
    plt.show()

def mpl_test():
    fig,ax=plt.subplots(figsize=(12,7))
    x=np.array(range(0,10,1))
    l=[i*i for i in x]
    y=np.array(l)
    l1=[i**3 for i in x]
    y2=np.array(l1)
    l3 = [i  for i in x]
    y3 = np.array(l3)
    ax.plot(x,y,color='red',linestyle='-',marker='*',label='парабола')
    ax.plot(x,y2,color='yellow',linestyle='-.',marker='o',label='кубическая')
    ax.plot(x, y3,color='black',linestyle=':',marker='^',label='линия')
    ax.set_ylim(0,50)
    ax.set_title('График')
    ax.set_xlabel('ось x')
    ax.set_ylabel('ось y')
    ax.set_xticks(np.arange(0,10,0.5))
    ax.set_yticks(np.arange(0, 50, 5))
    ax.legend(loc='lower right')
    fig.savefig('fig.jpg')
    plt.show()


def task1(param):
    path = '01_apc.tif'
    image = io.imread(path)

    fig, ax = plt.subplots(2,2, figsize=(12, 7))

    ax[0,0].imshow(image, cmap='gray')
    ax[0,0].set_title('Исходное изображение')
    hist, bin_centers = histogram(image,normalize=False)
    ax[1,0].plot(bin_centers,hist)
    ax[1, 0].set_title('Исходная гистограмма')

    #Пороговая обработка

    image[image < param]=0
    image[image >= param]=255

    ax[0,1].imshow(image, cmap='gray')
    ax[0,1].set_title('Изображение после пороговое бработки ')
    hist2, bin_centers2 = histogram(image,normalize=False)
    ax[1,1].plot(bin_centers2,hist2)
    ax[1, 1].set_title('Гистограмма после пороговой обработки')

    x=np.arange(0,256,1)
    y=x.copy()
    y[y<param]=0
    y[y>=param]=1
    fig_2, ax_2 = plt.subplots( figsize=(7, 4))
    ax_2.plot(x,y)
    ax_2.set_title('График функции пороговой обработки')
    plt.show()


def task2():
    path = '02_apss.tif'
    image = io.imread(path)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))


    ax[0, 0].set_title('Исходное изображение')
    ax[1, 0].set_title('Исходная гистограмма')
    ax[0, 0].imshow(image, cmap='gray')
    hist,bin_centers=histogram(image)
    ax[1,0].plot(bin_centers,hist)

    fmin=np.min(image)
    fmax=np.max(image)

    a=255/( fmax-fmin )
    b=-((255*fmin)/( fmax-fmin ))

    lineImage=a*image+b
    lineImage=lineImage.astype('uint8')

    ax[0, 1].set_title('Изображение после линейного конрастирования')
    ax[1, 1].set_title('Гистограмма после линейного контрастирования')
    ax[0, 1].imshow(lineImage, cmap='gray')
    hist, bin_centers = histogram(lineImage)
    ax[1, 1].plot(bin_centers, hist)

    x=np.arange(0,256,1)
    y = a*x+b
    fig_2, ax_2 = plt.subplots(figsize=(7, 3))
    ax_2.plot(x, y)
    ax_2.set_title('График функции линейного контрастирования')

    plt.show()



def task3():
    path = '01_apc.tif'

    fig, ax = plt.subplots(4, 2, figsize=(12, 7))

    image = io.imread(path)
    ax[0, 0].imshow(image, cmap='gray')
    ax[0,0].set_title('Стандартное изображение')


    image_hist, bins =histogram(image)
    bins_min=np.min(bins)
    bins_max=np.max(bins)

    start_hist=np.zeros(bins_min,dtype=np.int64)
    end_hist=np.zeros(255-bins_max,dtype=np.int64)

    start_bins=np.arange(0,bins_min,1)
    end_bins=np.arange(bins_max+1,256,1)

    image_hist=np.concatenate((start_hist,image_hist))
    image_hist=np.concatenate((image_hist,end_hist))

    bins = np.concatenate((start_bins, bins))
    bins = np.concatenate((bins, end_bins))

    print("image_hist=", image_hist)
    print()
    print("bins=", bins)

    h, b = np.histogram(image.flatten(), 256, [0, 256])

    ax[0,1].plot(bins,image_hist)
    ax[0,1].set_title('Гистограмма  Изо исходное')



    st_eq_image = equalize_hist(image)
    ax[1, 0].imshow(st_eq_image, cmap='gray')
    ax[1, 0].set_title('Еквализация Стандартная')
    hist_eq_image,bins_eq_im=histogram(st_eq_image)
    ax[1,1].plot(bins_eq_im,hist_eq_image)
    ax[1, 1].set_title('Гистограмма еквал. станд. ')


        #EКВАЛИЗАЦИЯ
    sum_f=image_hist.cumsum()
    F_f=(sum_f-sum_f.min())/(sum_f.max()-sum_f.min())
    g_f=255*F_f
    eq_image=g_f[image]

    ax[2,0].imshow(eq_image, cmap='gray')
    ax[2,0].set_title('Изо после эквализации самописная')

    eq_hist,eq_bins=histogram(eq_image)#ГИСТОГРАММА ЕКВАЛ.САМОПИСНАЯ
    ax[2,1].plot(eq_bins,eq_hist)
    ax[2, 1].set_title('Гистограмма еквал. cамописная')



    ax[3, 0].set_title('График функции поэлементоного преобразования')
    range_x=np.arange(0,256,1)
    ax[3,0].plot(range_x,g_f)

    hist_image_post, bins_i_past = np.histogram(eq_image.flatten(),256,[0,256])
    sum_g_post = hist_image_post.cumsum()
    F_g = (sum_g_post - sum_g_post.min()) / (sum_g_post.max() - sum_g_post.min())

    ax[3,1].plot(range_x,F_g)
    ax[3, 1].set_title('График инт-ой ф-и распр яркости ПОСЛЕ')


    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(range_x, F_f,color='blue')
    ax2.set_title('График интег-ой ф-и распр яркости ДО')

    plt.show()



if __name__ == '__main__':
    task1(75)
    task2()
    task3()










