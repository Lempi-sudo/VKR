import numpy as np
from skimage.io import imread , imshow , show
from matplotlib import pyplot as plt
from skimage import io
from PIL import Image, ImageDraw
from skimage.util import random_noise
from skimage.exposure import histogram, equalize_hist



def lsh_embed(C,b,seed):
    if seed<0:
        coords=np.array(range(0,len(b)))
    else:
        np.random.seed(seed)
        coords=np.array(range(0,C.size))
        np.random.shuffle(coords)
        coords=coords[0:len(b)]

    first_plane=C-2*(C//2) # C % 2
    new_first_plane=first_plane.copy()
    np.put(new_first_plane,coords,b)
    cw=C-first_plane+new_first_plane
    return  cw

def lsb_extract(Cw , Nb ,seed):
    if seed<0:
        coords=list(range(0,Nb))
    else:
        np.random.seed(seed)
        coords=np.array(range(0,Cw.size))
        np.random.shuffle(coords)
        coords=coords[0:Nb]
    first_plane=Cw%2
    Br=np.take(np.ravel(first_plane),coords)
    return Br

def bit_error_rate(b ,br):
    return np.sum(np.abs(b-br)) / b.size

C=imread("D:\Code\Python\WaterMarking\Image\goldhill.tif")
Nb=200
b=np.where(np.random.rand(Nb)>0.5,1,0)

seed=3


Cw=lsh_embed(C,b,seed)
br=lsb_extract(Cw,Nb,seed)
BER1=bit_error_rate(b,br)


seed=-1


Cw1=lsh_embed(C,b,seed)
br=lsb_extract(Cw1, Nb,seed)
BER2=bit_error_rate(b,br)
print(BER2)







