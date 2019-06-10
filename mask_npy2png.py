import os
import numpy as np
from PIL import Image


dir_npy = "data/output/mask/"
dir_out = "data/vis/mask_png/"

npy_name_list = [
    "0000_0072",
    "0000_0103",
    "0001_0136"
]

npy_path_list = []
for name in npy_name_list:
    npy_path_list.append(os.path.join(dir_npy, name) + '.npy')

for i in range(len(npy_path_list)):
    tensor_mask = np.load(npy_path_list[i])
    tensor_zero = np.zeros((1, tensor_mask.shape[1], tensor_mask.shape[2]))
    tensor_chw = np.concatenate((tensor_mask, tensor_zero), axis=0) * 255
    tensor_rgb = (np.transpose(tensor_chw, (1,2,0))).astype(np.uint8)
    mask_im = Image.fromarray(tensor_rgb)
    _name = npy_name_list[i] + '_vis.png'
    mask_im.save(os.path.join(dir_out, _name))
