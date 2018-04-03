import idx2numpy
import numpy as np
from __main__ import *

arr = np.zeros([3,3,3,3], dtype=np.float32)
tmp = 1.0

for i0 in range(0, arr.shape[0]):
    for i1 in range(0, arr.shape[1]):
        for i2 in range(0, arr.shape[2]):
            for i3 in range(0, arr.shape[3]):
                tmp = tmp * -0.5
                arr[i0, i1, i2, i3] = tmp
    tmp = 1.0

outpath = mkdir(TEST_DATA_DIR, 'idxImport')
out_file_name = str(outpath / 'float_4d_power2.idx')
f_write = open(out_file_name, 'wb')
idx2numpy.convert_to_file(f_write, arr)
f_write.close()

print("float_4d_power2 sum: ", np.sum(arr))

# print('data size: ', arr.size)
# print('expected output: ')
# print(arr.flatten())