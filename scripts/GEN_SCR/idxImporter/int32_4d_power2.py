import idx2numpy
import numpy as np
from __main__ import *

arr = np.zeros([3,3,3,3], dtype=np.int32)
tmp = 1

for i0 in range(0, arr.shape[0]):
    for i1 in range(0, arr.shape[1]):
        for i2 in range(0, arr.shape[2]):
            for i3 in range(0, arr.shape[3]):
                arr[i0, i1, i2, i3] = tmp - 1
                tmp = tmp * 4
                if tmp > 2**32:
                    tmp = 1


outpath = mkdir(TEST_DATA_DIR, 'idxImport')
out_file_name = str(outpath / 'int32_4d_power2.idx')
f_write = open(out_file_name, 'wb')
idx2numpy.convert_to_file(f_write, arr)
f_write.close()

print("int32_4d_power2 sum: ", np.sum(arr))

# print('data size: ', arr.size)
# print('expected output: ')
# print(arr.flatten())