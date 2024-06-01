
import numpy as np
import tensorflow as tf
import pyuTensor

pyuTensor.set_ram_total(2000000)
pyuTensor.set_meta_total(1000)

a_setting = (1, 5, 5, 3)
a = np.arange(np.prod(a_setting)).reshape(a_setting).astype(np.float32)
b_setting = (5, 5, 3, 5)
b = np.arange(np.prod(b_setting)).reshape(b_setting).astype(np.float32)

print(pyuTensor.conv2d_f(a, b.transpose(3, 0, 1, 2), [0 for _ in range(b_setting[3])], [1, 2, 2, 1], "VALID"))
print(tf.nn.conv2d(a, b, strides=[1, 2, 2, 1], padding="VALID").numpy())