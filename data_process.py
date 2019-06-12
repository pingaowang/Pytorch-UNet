import os
import numpy as np
from PIL import Image
from utils import int01_3darr_save2png


def int01_2darr_save2png(arr, out_path):
    im = Image.fromarray((arr * 255).astype(np.uint8))
    im.save(out_path)


def get_img_cls_from_path(img_path):
    return int(img_path[-5])


def get_img_id(img_path, type='str'):
    assert type in ['str', 'int']
    if type == 'str':
        return img_path[-10:-6]
    elif type == 'int':
        return int(img_path[-10:-6])


def get_arr_int_padded(path_img):
    """
    Not robust, only used for image size [13200, 10200]
    :param path_img: path of the png image
    :return: int[0, 1] zero padded np array, size: 12500, 10500]
    """
    arr_img = np.array(Image.open(path_img))
    # convert type to int
    if len(arr_img.shape) == 3:
        arr_int = (np.mean(arr_img, 2) - 255).astype(bool).astype(int)
    elif len(arr_img.shape) == 2:
        arr_int = 1 - arr_img.astype(bool).astype(int)
    else:
        assert len(arr_img.shape) == 2, "img shape error, {}".format(len(arr_img.shape) == 2)
    # zero padding
    arr_int_padded = np.pad(arr_int, ((300, 0), (0, 300)), 'constant')
    return arr_int_padded


def generate_unit_tensors(tensor_all, mask_out_dir, png_out_dir, img_id):
    """
    processing the large mask tensor to unit mask tensors, and save: 1) mask npy 2)input png.
    :param tensor_all:
    :param out_dir:
    :param img_id:
    :return:
    """
    len_dim0 = tensor_all.shape[1]
    len_dim1 = tensor_all.shape[2]
    assert len_dim0 % side_length == 0
    assert len_dim1 % side_length == 0

    count = 0
    count_1 = 0
    count_0 = 0
    stride = 250
    list_stride = []
    for i in range(side_length // stride):
        list_stride.append(i * stride)

    print("===========")
    print("Generating image: " + img_id)
    for each_stride in list_stride:
        print(" Processing stride={} ...".format(each_stride))
        for i in range(int(len_dim0 / side_length)):
            for j in range(int(len_dim1 / side_length)):
                arr_unit = tensor_all[:, # dim 0: cls
                           i * side_length + each_stride: i * side_length + side_length + each_stride, # dim 1: raw
                           j * side_length + each_stride: j * side_length + side_length + each_stride]# dim 2: column
                if np.sum(arr_unit, axis=(0, 1, 2)) == 0:
                    count_0 += 1
                else:
                    # 1) save mask npy
                    # out npy naming rule: [img_id, block_id]
                    npy_out_name = img_id + '_{0:04d}'.format(count_1)
                    npy_out_path = os.path.join(mask_out_dir, npy_out_name)
                    np.save(npy_out_path, arr_unit)
                    print("Saved mask: " + npy_out_path)
                    # 2) save png
                    # build png tensor
                    int_tensor_png = np.sum(arr_unit, axis=0).astype(bool).astype(int)
                    # save tensor to png
                    # png naming rule: [img_id, block_id]
                    png_out_name = img_id + '_{0:04d}'.format(count_1) + '.png'
                    png_out_path = os.path.join(png_out_dir, png_out_name)
                    int01_2darr_save2png(int_tensor_png, out_path=png_out_path)
                    print("Saved input png: " + png_out_path)

                    count_1 += 1
                count += 1

    print("Finished. Logs:")
    print("There are totally {} blocks.".format(count))
    print("There are totally {} all-zero blocks.".format(count_0))
    print("There are totally {} non-zero blocks.".format(count_1))


# Example: Save to black and white png
# int01_2darr_save2png(arr_padded, out_dir='data/a_real_png_sample/out_2.png')

# Example: Save to npy
# np.save('data/a_real_png_sample/out', arr_padded)


if __name__ == '__main__':
    side_length = 500
    n_classes = 3
    mask_dir = 'data/input_raw_images'
    mask_out_dir = 'data/data_proc_output/mask'
    png_out_dir = 'data/data_proc_output/png'
    if not os.path.isdir(mask_out_dir):
        os.mkdir(mask_out_dir)
    if not os.path.isdir(png_out_dir):
        os.mkdir(png_out_dir)

    # build class list: [0, 1, ...], int
    list_class = []
    for i in range(n_classes):
        list_class.append(i)

    # images' path
    list_img_path = os.listdir(mask_dir)
    for i in list_img_path:
        assert os.path.isfile(os.path.join(mask_dir, i)), "image doesn't exist: {}".format(i)
        if '.DS_Store' in list_img_path:
            list_img_path.remove('.DS_Store')
    # show the list of images
    print(list_img_path)

    # build image_id list
    list_image_id = []
    for img_path in list_img_path:
        img_id_str = get_img_id(img_path, 'str')
        if img_id_str not in list_image_id:
            list_image_id.append(img_id_str)

    # 1) Read mask images.
    # 2) Save unit mask and png.
    for image_id_str in list_image_id:
        list_arr_int_padded = []
        for cls_int in list_class:
            img_name = image_id_str + '_' + str(cls_int) + '.png'
            full_img_path = os.path.join(mask_dir, img_name)
            # build mask 2d arr for each image:
            arr_int_padded = get_arr_int_padded(full_img_path)
            list_arr_int_padded.append(arr_int_padded)
        # build mask tensor:
        tensor_all = np.stack(tuple(list_arr_int_padded))

        generate_unit_tensors(tensor_all=tensor_all,
                              mask_out_dir=mask_out_dir,
                              png_out_dir=png_out_dir,
                              img_id=image_id_str)















