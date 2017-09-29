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

with tf.Session() as sess:


    a = tf.cast(tf.convert_to_tensor(np.random.rand(10,10) * 2), tf.float32)
    b = tf.cast(tf.convert_to_tensor(np.random.rand(10,10) * 2), tf.float32)

    a_max = tf.reduce_max(a)
    a_min = tf.reduce_min(a)
    [q_a, a_min, a_max] = gen_array_ops.quantize_v2(a, a_min, a_max, tf.quint8, "MIN_FIRST")


    b_max = tf.reduce_max(b)
    b_min = tf.reduce_min(b)
    [q_b, b_min, b_max] = gen_array_ops.quantize_v2(b, b_min, b_max, tf.quint8, "MIN_FIRST")

    # a = tf.cast(a, tf.uint8)
    # a = tf.bitcast(a, tf.quint8)
    # b = tf.cast(b, tf.uint8)
    # b = tf.bitcast(b, tf.quint8)

    [q_out, min_out, max_out] = gen_math_ops.quantized_mat_mul(q_a, q_b, a_min, a_max, b_min, b_max, Toutput=tf.qint32)

    [r_min_out, r_max_out] = gen_math_ops.requantization_range(q_out, min_out, max_out)

    [rq_out, q_min, q_max] = gen_math_ops.requantize(q_out, min_out, max_out, r_min_out, r_max_out, tf.quint8)

    dq_out = gen_array_ops.dequantize(rq_out, q_min, q_max, "MIN_FIRST")

    reference_out = tf.matmul(a, b)
    diff = tf.div(reference_out, dq_out)
    
    print("A")
    print(a.eval())
    print("B")
    print(b.eval())
    print("C")

    print("Qantized C")
    print(dq_out.eval())
    print("min_out", q_min.eval(), ",", "max_out", q_max.eval())
    print("diff:")
    print(diff.eval())
