from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
import tensorflow as tf
import numpy as np

#Where this source lies...
#from tensorflow.python.ops import gen_math_ops
#ipython3: gen_math_ops?? 
#file: gen_math_op.py -> def quantized_mat_mul -> _op_def_lib.apply_op("QuantizedMatMul"...
#file: .../core/kernels/quantized_matmul_op.cc -> REGISTER_KERNEL_BUILDER(Name("QuantizedMatMul")... -> ReferenceGemm
#file: .../core/kernels/reference_gemm.h -> ReferenceGemm()

#file /usr/local/lib/python3.6/site-packages/tensorflow/python/ops/gen_array_ops.py

with tf.Session() as sess:


    a = tf.cast(tf.convert_to_tensor(np.random.rand(1024,1024) * 10), tf.float32)
    b = tf.cast(tf.convert_to_tensor(np.random.rand(1024,1) * 10), tf.float32)
    # a = tf.cast(tf.convert_to_tensor(np.ones((1024,1024)) * 10), tf.float32)
    # b = tf.cast(tf.convert_to_tensor(np.ones((1024,1)) * 10), tf.float32)

    a_min = tf.reduce_min(a)
    a_max = tf.reduce_max(a)
    [q_a, a_min, a_max] = gen_array_ops.quantize_v2(a, a_min, a_max, tf.quint8, "MIN_FIRST", name="qA")
    
    b_min = tf.reduce_min(b)
    b_max = tf.reduce_max(b)
    [q_b, b_min, b_max] = gen_array_ops.quantize_v2(b, b_min, b_max, tf.quint8, "MIN_FIRST", name="qB")

    print("------- float32 input matrices ------")
    print("a min: ", a_min.eval(), " a max: ", a_max.eval())
    print("b min: ", b_min.eval(), " b max: ", b_max.eval())

    # a = tf.cast(a, tf.uint8)
    # a = tf.bitcast(a, tf.quint8)
    # b = tf.cast(b, tf.uint8)
    # b = tf.bitcast(b, tf.quint8)

    min_out = 0
    max_out = 0

    [q_out, min_out, max_out] = gen_math_ops.quantized_mat_mul(q_a, q_b, a_min, a_max, b_min, b_max, Toutput=tf.qint32, name="qMatMul")

    print("------- quantized_mat_mul ------")
    print("min: ", min_out.eval(), " max: ", max_out.eval(), "mean: ", tf.reduce_mean(tf.bitcast(q_out, tf.int32)).eval())

    [request_min_out, request_max_out] = gen_math_ops.requantization_range(q_out, min_out, max_out, name="rqRange")

    print("------- requantization_range ------")
    print("min: ", request_min_out.eval(), " max: ", request_max_out.eval())

    [rq_out, rq_min_out, rq_max_out] = gen_math_ops.requantize(q_out, min_out, max_out, request_min_out, request_max_out, tf.quint8, name="rQ")

    print("------- requantize ------")
    print("min: ", rq_min_out.eval(), " max: ", rq_max_out.eval(), "mean: ", tf.reduce_mean(tf.to_float(tf.bitcast(rq_out, tf.uint8))).eval())

    dq_out = gen_array_ops.dequantize(rq_out, rq_min_out, rq_max_out, "MIN_FIRST", name="deQ")

    reference_out = tf.matmul(a, b)
    diff = tf.subtract(reference_out, dq_out)
    diff = tf.reduce_mean(tf.abs(diff))         #average delta per element
    
    print("------- dequantize ------")
    print("mean: ", tf.reduce_mean(tf.bitcast(tf.reduce_mean(dq_out), tf.uint8)).eval())
    print("diff: ", diff.eval(), ", percent diff: ", diff.eval() / tf.reduce_mean(reference_out).eval() * 100, "%")

