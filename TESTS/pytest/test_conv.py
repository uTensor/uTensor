def test_conv(random_conv_input):
    import numpy as np
    import pyuTensor
    import tensorflow as tf

    pyuTensor.set_ram_total(1000 * 1024)
    pyuTensor.set_meta_total(100 * 1024)

    for input_tensor, filter_tensor, bias, strides, padding in random_conv_input:
        uT_res = pyuTensor.conv2d(
            input_tensor, filter_tensor.transpose(3, 0, 1, 2), bias=bias, strides=strides, padding=padding,
        )
        tf_res = (
            tf.nn.conv2d(
                input_tensor, filter_tensor, strides=strides, padding=padding
            ).numpy()
            + bias
        )

        assert np.allclose(uT_res, tf_res)

def test_maxpool(random_4d_input):
    from random import choice, randint

    import numpy as np
    import pyuTensor
    import tensorflow as tf

    pyuTensor.set_ram_total(1000 * 1024)
    pyuTensor.set_meta_total(100 * 1024)

    for input_tensor in random_4d_input:
        padding = choice(["VALID", "SAME"])
        h, w = input_tensor.shape[1:3]
        k_h, k_w = randint(1, h), randint(1, w)
        uT_res = pyuTensor.max_pool2d(input_tensor, [k_h, k_w], [1, 2, 2, 1], padding)
        tf_res = tf.nn.max_pool2d(
            input_tensor, ksize=[1, k_h, k_w, 1], strides=[1, 2, 2, 1], padding=padding
        ).numpy()

        assert np.allclose(uT_res, tf_res)

        uT_res = pyuTensor.max_pool2d(input_tensor, [k_h, k_w], [1, 1, 1, 1], padding)
        tf_res = tf.nn.max_pool2d(
            input_tensor, ksize=[1, k_h, k_w, 1], strides=[1, 1, 1, 1], padding=padding
        ).numpy()
        assert np.allclose(uT_res, tf_res)