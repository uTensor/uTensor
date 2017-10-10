from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import numpy as np

with tf.Session() as sess:

    rand_input = tf.convert_to_tensor(np.random.rand(28,28), dtype=tf.float32)
    new_dim = tf.convert_to_tensor(np.array([784,1]), dtype=tf.int32)
    reshaped_tensor = tf.reshape(rand_input, new_dim, name="ref_reshape")

    rand_act = tf.convert_to_tensor(np.random.rand(784,1) * 10, dtype=tf.float32)
    act_min = tf.reduce_min(rand_act, name="ref_min")
    act_max = tf.reduce_max(rand_act, name="ref_max")
    [q_a, a_min, a_max] = gen_array_ops.quantize_v2(rand_act, act_min, act_max, tf.quint8, "MIN_FIRST")
    [out_a, out_min, out_max] = gen_nn_ops.quantized_relu(q_a, a_min, a_max, tf.quint8, name="ref_qRelu")

    rand_a = tf.convert_to_tensor(np.random.rand(200,1) * 10, dtype=tf.float32)
    rand_b = tf.convert_to_tensor(np.random.rand(200,1) * 10, dtype=tf.float32)
    ab_sum = tf.add(rand_a, rand_b, name="ref_add")

    #max_i = math_ops.argmax(rand_act, 1, output_type=tf.int32, name="ref_argmax")

    #the data for max and min functions will be extracted from deep_mlp directly
