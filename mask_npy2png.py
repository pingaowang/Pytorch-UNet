import os
import numpy as np
from PIL import Image
from utils import int01_3darr_save2png, int01_3darr_save2png_3cls, listdir_check


dir_npy = "data/data_proc_output/mask/"
dir_out = "data/data_proc_output/mask_vis/"

# load npy
npy_file_list = os.listdir(dir_npy)
listdir_check(npy_file_list)

# get npy names
npy_path_list = []
for name in npy_file_list:
    npy_path_list.append(os.path.join(dir_npy, name))

# process and save to png
for i in range(len(npy_path_list)):
    tensor_mask = np.load(npy_path_list[i])

    _name = npy_file_list[i][:-4] + '_vis.png'
    out_path = os.path.join(dir_out, _name)

    int01_3darr_save2png_3cls(tensor_mask, out_path)



