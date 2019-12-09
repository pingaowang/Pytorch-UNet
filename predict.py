import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import cv2

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf, int01_3darr_save2png_3cls, listdir_check
from utils import plot_img_and_mask
from shutil import copyfile

from torchvision import transforms


N_CLASSES_AREA = 8
N_CLASSES_LINE = 4

# color the mask
# l_color = [
#     [25, 0, 0],
#     [50, 0, 0],
#     [75, 0, 0],
#     [100, 0, 0],
#     [0, 25, 0],
#     [0, 50, 0],
#     [0, 75, 0],
#     [0, 100, 0],
#     [0, 0, 25],
#     [0, 0, 50],
#     [0, 0, 75],
#     [0, 0, 100]
# ]

l_color = [
    [100, 50, 50],
    [50, 100, 50],
    [50, 50, 100],
    [100, 100, 50],
    [100, 100, 100],
    [120, 100, 100],
    [100, 120, 100],
    [100, 100, 120],
    [120, 120, 100],
    [120, 100, 120],
    [100, 120, 120],
    [120, 120, 120]
]


def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    # img = resize_and_crop(full_img, scale=scale_factor)
    # img = normalize(img)
    img = np.array(full_img, dtype=np.float32)
    img = normalize(img)

    # left_square, right_square = split_img_into_squares(img)
    #
    # left_square = hwc_to_chw(left_square)
    # right_square = hwc_to_chw(right_square)
    square = hwc_to_chw(img)
    #
    # X_left = torch.from_numpy(left_square).unsqueeze(0)
    # X_right = torch.from_numpy(right_square).unsqueeze(0)
    img = torch.from_numpy(square).unsqueeze(0)
    
    if use_gpu:
        # X_left = X_left.cuda()
        # X_right = X_right.cuda()
        img = img.cuda()

    with torch.no_grad():
        # output_left = net(X_left)
        # output_right = net(X_right)
        output = net(img)

        # left_probs = output_left.squeeze(0)
        # right_probs = output_right.squeeze(0)
        # probs = output # .squeeze(0)
        # arr_probs = np.array(probs.cpu())
        # arr_probs = np.transpose

        # tf = transforms.Compose(
        #     [
        #         # transforms.ToPILImage(),
        #         # transforms.Resize(img_height),
        #         transforms.ToTensor()
        #     ]
        # )
        
        # left_probs = tf(left_probs.cpu())
        # right_probs = tf(right_probs.cpu())
        # probs = tf(arr_probs)

        # left_mask_np = left_probs.squeeze().cpu().numpy()
        # right_mask_np = right_probs.squeeze().cpu().numpy()
        mask_np = output.squeeze().cpu().numpy()

    # full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    if use_dense_crf:
        # full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), mask_np)

    # return full_mask > out_threshold
    return mask_np > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='dir path of input images', required=True)

    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        help='dir path of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)
    parser.add_argument('--task-type', type=str, help='line or area')
    print(parser.parse_args())

    return parser.parse_args()

# def get_output_filenames(args):
#     in_files = args.input
#     out_files = []
#
#     if not args.output:
#         for f in in_files:
#             pathsplit = os.path.splitext(f)
#             out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
#     elif len(in_files) != len(args.output):
#         print("Error : Input files and output files are not of the same length")
#         raise SystemExit()
#     else:
#         out_files = args.output
#
#     return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def data_loader(in_dir, out_dir):
    # load in_files
    list_i_name = os.listdir(in_dir)
    listdir_check(list_i_name)
    list_i = []
    list_o = []
    for i_name in list_i_name:
        list_o.append(os.path.join(out_dir, 'predicted_png', i_name))
        list_i.append(os.path.join(in_dir, i_name))

    return list_i, list_o, list_i_name


def color_bin_arr(arr_bin, color):
    h, w = arr_bin.shape

    arr_colored = np.zeros((h, w, 3))

    arr_colored[:, :, 0] = arr_bin * color[0]
    arr_colored[:, :, 1] = arr_bin * color[1]
    arr_colored[:, :, 2] = arr_bin * color[2]

    return arr_colored


if __name__ == "__main__":
    args = get_args()

    assert args.task_type in ['line', 'area']
    if args.task_type == 'line':
        N_CLASSES = N_CLASSES_LINE
    elif args.task_type == 'area':
        N_CLASSES = N_CLASSES_AREA

    folder_predicted_png = os.path.join(args.output, 'predicted_png')
    folder_comparison = os.path.join(args.output, 'comparison_png')
    folder_full_pred = os.path.join(args.output, 'full_pred')
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    if not os.path.isdir(folder_predicted_png):
        os.mkdir(folder_predicted_png)
    if not os.path.isdir(folder_comparison):
        os.mkdir(folder_comparison)
    if not os.path.isdir(folder_full_pred):
        os.mkdir(folder_full_pred)

    in_files, out_files, l_img_names = data_loader(args.input, args.output)

    # copyfiles in input root
    in_root = args.input[:-4]
    for f_png_root in os.listdir(in_root):
        if f_png_root.endswith('.png'):
            copyfile(
                os.path.join(in_root, f_png_root),
                os.path.join(args.output, f_png_root)
            )

    net = UNet(n_channels=3, n_classes=N_CLASSES)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        # assert img.mode == 'RGB', "image is not RBG but is '{}'.".format(img.mode)

        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        img_size = mask.shape[1]
        arr_bin = mask.astype(int)
        arr_bin = np.transpose(arr_bin, (1, 2, 0))
        arr_color = np.zeros((img_size, img_size, 3))
        for i_cls in range(arr_bin.shape[2]):
        # for i_cls in range(3):
            arr_color_2 = color_bin_arr(arr_bin[:, :, i_cls], l_color[i_cls])
            arr_color = arr_color + arr_color_2
        arr_color = - (arr_color - 255)

        # if args.viz:
        #     print("Visualizing results for image {}, close to continue ...".format(fn))
        #     plot_img_and_mask(img, mask)

        ## Comparison of img and arr_color
        arr_img = np.array(img)
        arr_comb = np.concatenate((arr_img, arr_color), axis=1)

        if not args.no_save:
            # save pred_block
            out_fn = out_files[i]
            cv2.imwrite(out_fn, arr_color)

            # save comparison
            cv2.imwrite(os.path.join(folder_comparison, l_img_names[i]), arr_comb)

            print("Mask saved to {}".format(out_files[i]))


    """
    Combine output blocks to a fullsize image
    """
    N_ROW = 6
    N_COL = 6

    folder_png = folder_predicted_png
    file_out = os.path.join(folder_full_pred, 'fullsize.png')

    # prepare list of png filename
    l_filename = []
    for f_png in os.listdir(folder_png):
        if f_png.endswith('.png'):
            l_filename.append(f_png)
    l_filename.sort()
    assert N_ROW * N_COL == len(l_filename), "N_ROW * N_COL expected equal to len(l_filename)."

    h, w, c = cv2.imread(os.path.join(folder_png, l_filename[0])).shape
    h_full = N_COL * h
    w_full = N_ROW * w

    img_out = np.zeros((w_full, h_full, 3))

    for i in range(N_ROW):
        for j in range(N_COL):
            block = cv2.imread(os.path.join(folder_png, l_filename[j * N_COL + i]))
            img_out[w * i: w * (i + 1), h * j: h * (j + 1), :] = block

    # Save img_out
    cv2.imwrite(file_out, img_out)












