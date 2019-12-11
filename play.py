import numpy as np
from PIL import Image

arr = np.load('data/dataset_line_v2_test_10_2/mask/10_1_0008.npy')
h, w, c = arr.shape
arr_2 = np.zeros((h, w, 3))

for i in range(c):
    x, y = np.where(arr[:, :, i])
    arr_2[x, y, 0] = arr[x, y, i] * (i + 1) * 50

arr_2 = arr_2.astype(np.uint8)

print("# if all zeros:")
print(arr.sum())
print(arr_2.sum())
print("====")
print(np.unique(arr))
print(np.unique(arr_2))
print("====")

im = Image.fromarray(arr_2, 'RGB')
im.save('test_1.png')

arr_3 = np.array(im)[0]
print(arr_3.sum())
print(np.unique(arr_3))

im.show()