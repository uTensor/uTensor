def test_conv(random_conv_input):
    import pyuTensor
    import numpy as np
    import tensorflow as tf

    pyuTensor.set_ram_total(1000 * 1024)
    pyuTensor.set_meta_total(100 * 1024)

    for input_tensor, filter_tensor, bias, strides, padding in random_conv_input:
        uT_res = pyuTensor.conv2d_f(
            input_tensor, filter_tensor.transpose(3, 0, 1, 2), bias, strides, padding
        )
        tf_res = (
            tf.nn.conv2d(
                input_tensor, filter_tensor, strides=strides, padding=padding
            ).numpy()
            + bias
        )

        assert np.allclose(uT_res, tf_res)
