import numpy as np
from PIL import Image
import torchvision
from random import randrange
import random


N_CLS = 4

arr = np.load('data/dataset_line_v2_test_10_2/mask/10_1_0008.npy')
h, w, c = arr.shape
arr_2 = np.zeros((h, w, 3))

for i in range(c):
    x, y = np.where(arr[:, :, i])
    arr_2[x, y, 0] = arr[x, y, i] * (i + 1) * 50

arr_2 = arr_2.astype(np.uint8)

im = Image.fromarray(arr_2, 'RGB')

seed = np.random.randint(12452346)

# resize, crop, ratio
random.seed(seed)
trans_1 = torchvision.transforms.RandomResizedCrop(size=(500), interpolation=Image.NEAREST)
im_4 = trans_1(im)
im_4.show()

# rotate
random_degree = randrange(360)
im_5 = torchvision.transforms.functional.rotate(im, angle=random_degree)  #, interpolation=Image.NEAREST) , angle=random_degree
im_5.show()

np.unique(np.array(im_5))
im.show()