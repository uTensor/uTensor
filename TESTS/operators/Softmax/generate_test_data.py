import numpy as np

import tensorflow as tf
from utensor_cgen.utils import save_idx

graph = tf.Graph()
with graph.as_default():
    tf_rand_arr = tf.constant(np.random.rand(10, 5), dtype=tf.float32)
    tf_softmax = tf.nn.softmax(tf_rand_arr, name="softmax")

with tf.Session(graph=graph) as sess:
    np_arr = tf_rand_arr.eval()
    np_softmax = tf_softmax.eval()

save_idx(np_arr, "TESTS/constants/in/logits.idx")
save_idx(np_softmax, "TESTS/constants/out/ref_sortmax.idx")
