import random
import numpy as np
from PIL import Image


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    # img = pilimg.resize((newW, newH))
    # img = img.crop((0, diff // 2, newW, newH - diff // 2))
    img = pilimg.crop((0, diff // 2, newW, newH - diff // 2))

    np_img = np.array(img, dtype=np.float32)
    if len(np_img.shape)==2:
        np_img = np.expand_dims(np_img, 2)

    return np_img

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return - (x / 255 - 0.5)

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def int01_2darr_save2png(arr, out_path):
    assert len(arr.shape) == 2, "input # dim expected 2, but here is {}".format(len(arr.shape))

    im = Image.fromarray((arr * 255).astype(np.uint8))
    im.save(out_path)


def int01_3darr_save2png(arr, out_path):
    assert len(arr.shape) == 3, "input # dim expected 3, but here is {}".format(len(arr.shape))
    assert arr.shape[0] == 2, "Currently only used for 2 classes."

    tensor_zero = np.zeros((1, arr.shape[1], arr.shape[2]))
    tensor_chw = np.concatenate((arr, tensor_zero), axis=0) * 255
    tensor_rgb = (np.transpose(tensor_chw, (1, 2, 0))).astype(np.uint8)
    Image.fromarray(tensor_rgb).save(out_path)


def int01_3darr_save2png_3cls(arr, out_path):
    assert len(arr.shape) == 3, "input # dim expected 3, but here is {}".format(len(arr.shape))
    assert arr.shape[0] == 3, "Currently only used for 3 classes, but here is {}".format(arr.shape[0])

    tensor_chw = arr * 255
    tensor_rgb = (np.transpose(tensor_chw, (1, 2, 0))).astype(np.uint8)
    Image.fromarray(tensor_rgb).save(out_path)


def listdir_check(l):
    if '.DS_Store' in l:
        l.remove('.DS_Store')