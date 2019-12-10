import numpy as np
from PIL import Image

arr = np.load('data/dataset_line_v2_test_10_2/mask/10_1_0008.npy')
h, w, c = arr.shape
arr_2 = np.zeros((h, w, 3))

for i in range(c):
    arr_2[:, :, 0] = arr_2[:, :, 0] + arr[:, :, i]

arr_2 = arr_2 * 10
arr_2 = arr_2.astype(np.uint8)

print("# if all zeros:")
print(arr.sum())
print(arr_2.sum())
print("====")
print(np.unique(arr))
print(np.unique(arr_2))
print("====")

im = Image.fromarray(arr_2, 'RGB')
im.show()

arr_3 = np.array(im)
print(arr_3.sum())
print(np.unique(arr_3))

im.show()