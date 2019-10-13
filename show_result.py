import cv2
import os
from utils import listdir_check
import numpy as np


ori_dir = "data/0010_for_test_3cls/mask_vis/"
list_pred_dir = [
    # "data/test_6_cp1/",
    # "data/test_6_cp2/",
    # "data/test_6_cp3/",
    # "data/test_6_cp4/",
    "data/test_6_cp5/"
]

assert len(list_pred_dir) < 5, "Corrently only for 5 prediced images"

list_names = []

list_ori_name = os.listdir(ori_dir)
listdir_check(list_ori_name)

for ori_name in list_ori_name:
    list_names.append(ori_name[:9])

# Compare 5 groups of predicted images with the original image.
if len(list_pred_dir) == 5:
    for i in range(len(list_ori_name)):
        png_ori = os.path.join(ori_dir, list_names[i] + '_vis.png')

        list_png_pred = []
        for pred_dir in list_pred_dir:
            png_pred = os.path.join(pred_dir, 'pred_' + list_names[i] + '.png')
            list_png_pred.append(png_pred)

        # the original image
        image_1 = cv2.imread(png_ori)
        image_1 = cv2.resize(image_1, (250, 250))

        # predicted images
        list_pred_images = [image_1]
        for png_pred in list_png_pred:
            image_2 = cv2.imread(png_pred)
            image_2 = cv2.resize(image_2, (250, 250))
            list_pred_images.append(image_2)

        numpy_horizontal_1 = np.hstack(tuple(list_pred_images[:2]))
        numpy_horizontal_2 = np.hstack(tuple(list_pred_images[2:4]))
        numpy_horizontal_3 = np.hstack(tuple(list_pred_images[4:]))
        numpy_vertical = np.vstack((numpy_horizontal_1, numpy_horizontal_2, numpy_horizontal_3))

        cv2.imshow("id: {}".format(list_ori_name[i]), numpy_vertical)

        cv2.waitKey()

# Compare 3 groups of predicted images with the original image.
elif len(list_pred_dir) == 3:
    for i in range(len(list_ori_name)):
        png_ori = os.path.join(ori_dir, list_names[i] + '_vis.png')

        list_png_pred = []
        for pred_dir in list_pred_dir:
            png_pred = os.path.join(pred_dir, 'pred_' + list_names[i] + '.png')
            list_png_pred.append(png_pred)

        # the original image
        image_1 = cv2.imread(png_ori)
        image_1 = cv2.resize(image_1, (250, 250))

        # predicted images
        list_pred_images = [image_1]
        for png_pred in list_png_pred:
            image_2 = cv2.imread(png_pred)
            image_2 = cv2.resize(image_2, (250, 250))
            list_pred_images.append(image_2)

        numpy_horizontal_1 = np.hstack(tuple(list_pred_images[:2]))
        numpy_horizontal_2 = np.hstack(tuple(list_pred_images[2:]))
        numpy_vertical = np.vstack((numpy_horizontal_1, numpy_horizontal_2))

        cv2.imshow("id: {}".format(list_ori_name[i]), numpy_vertical)

        cv2.waitKey()

# Compare 1 group of predicted images with the original image.
elif len(list_pred_dir) == 1:
    for i in range(len(list_ori_name)):
        png_ori = os.path.join(ori_dir, list_names[i] + '_vis.png')

        list_png_pred = []
        for pred_dir in list_pred_dir:
            png_pred = os.path.join(pred_dir, 'pred_' + list_names[i] + '.png')
            list_png_pred.append(png_pred)

        # the original image
        image_1 = cv2.imread(png_ori)

        # predicted images
        list_pred_images = [image_1]
        for png_pred in list_png_pred:
            image_2 = cv2.imread(png_pred)
            list_pred_images.append(image_2)

        numpy_horizontal_1 = np.hstack(tuple(list_pred_images))

        cv2.imshow("id: {}".format(list_ori_name[i]), numpy_horizontal_1)

        cv2.waitKey()