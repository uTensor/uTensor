def test_conv():
    import _pyuTensor
    import numpy as np
    import tensorflow as tf

    _pyuTensor.set_ram_total(4096)
    _pyuTensor.set_meta_total(4096)

    # input: B H W C
    a_setting = (1, 5, 5, 3)
    a = np.arange(np.prod(a_setting)).reshape(a_setting).astype(np.float32)
    # filter: H W Cin Cout
    b_setting = (5, 5, 3, 5)
    b = np.arange(np.prod(b_setting)).reshape(b_setting).astype(np.float32)

    uT_1 = _pyuTensor.conv2d_f(a, b.transpose(3, 0, 1, 2), [0 for _ in range(b_setting[3])], [1, 2, 2, 1], "VALID")
    uT_2 = _pyuTensor.conv2d_f(a, b.transpose(3, 0, 1, 2).copy(), [0 for _ in range(b_setting[3])], [1, 2, 2, 1], "VALID")
    tf_1 = tf.nn.conv2d(a, b, strides=[1, 2, 2, 1], padding="VALID").numpy()

    assert np.allclose(uT_1, tf_1)
    assert np.allclose(uT_2, tf_1)