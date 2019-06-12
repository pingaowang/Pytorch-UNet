import os
import cv2
import numpy as np
from PIL import Image
from utils import listdir_check, int01_2darr_save2png


def combin_png(l_png, out_dir):
    # Get name
    png_name = l_png[0][:4]
    for png in l_png:
        # Check if they have same name
        assert png_name == png[:4], "PNG files from one case should share same names. " \
                                "here {} and {} are different.".format(name, png[:4])
        # Check if suffix is '.png'
        assert png[-4:] == '.png', "Suffix expected .png, but here is {}".format(png[-4:])

    list_bool_arr = []
    for png in l_png:
        # convert png to cv2 img instance
        im = Image.open(os.path.join(dir_raw, png))
        arr_rgb = np.array(im)
        # convert to numpy arr bool
        arr_bool_2d = np.sum((arr_rgb - 255), axis=2).astype(bool)

        list_bool_arr.append(arr_bool_2d)

    # combine
    arr_combine_int_3d = np.stack(tuple(list_bool_arr)).astype(int)
    arr_combine_int_2d = np.sum(arr_combine_int_3d, 0)
    arr_combine_bool_2d = arr_combine_int_2d.astype(bool)
    arr_combine_int01_2d = 1 - (arr_combine_bool_2d.astype(int))

    # save png
    out_path = os.path.join(out_dir, png_name + '_2.png')
    int01_2darr_save2png(arr_combine_int01_2d, out_path)


dir_raw = "data/combine_equ_raw"
out_dir = "data/combined_equ_png"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

l_files = os.listdir(dir_raw)
listdir_check(l_files)

l_names = []
for f in l_files:
    if f[:4] not in l_names:
        l_names.append(f[:4])

for name in l_names:
    l_file_each_name = []
    for f in l_files:
        if name in f:
            l_file_each_name.append(f)
    combin_png(l_file_each_name, out_dir)
