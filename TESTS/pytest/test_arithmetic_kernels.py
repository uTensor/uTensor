from typing import List

def test_broadcast_add(random_broadcastable_shapes: List[tuple]):
    import pyuTensor
    import numpy as np
    import tensorflow as tf

    pyuTensor.set_ram_total(100*1024)
    pyuTensor.set_meta_total(100*1024)
    for shape_a, shape_b in random_broadcastable_shapes:
        a = np.arange(np.prod(shape_a)).reshape(shape_a).astype(np.float32)
        b = np.arange(np.prod(shape_b)).reshape(shape_b).astype(np.float32)

        uT_1 = pyuTensor.add_kernel(a, b)
        np_1 = a + b
        tf_1 = tf.add(a, b).numpy()

        assert np.allclose(np_1, tf_1)
        assert np.allclose(uT_1, tf_1)

def test_broadcast_mul(random_broadcastable_shapes: List[tuple]):
    import pyuTensor
    import numpy as np
    import tensorflow as tf

    pyuTensor.set_ram_total(100*1024)
    pyuTensor.set_meta_total(100*1024)
    for shape_a, shape_b in random_broadcastable_shapes:
        a = np.arange(np.prod(shape_a)).reshape(shape_a).astype(np.float32)
        b = np.arange(np.prod(shape_b)).reshape(shape_b).astype(np.float32)

        uT_1 = pyuTensor.mul_kernel(a, b)
        np_1 = a * b
        tf_1 = tf.multiply(a, b).numpy()

        assert np.allclose(np_1, tf_1)
        assert np.allclose(uT_1, tf_1)

def test_add():
    import pyuTensor
    import numpy as np
    import tensorflow as tf

    pyuTensor.set_ram_total(4096)
    pyuTensor.set_meta_total(4096)

    a_setting = (5, 5, 3)
    a = np.arange(np.prod(a_setting)).reshape(a_setting).astype(np.float32)
    b_setting = a_setting
    b = np.arange(np.prod(b_setting)).reshape(b_setting).astype(np.float32)

    uT_1 = pyuTensor.add_kernel(a, b)
    np_1 = a + b
    tf_1 = tf.add(a, b).numpy()

    assert np.allclose(np_1, tf_1)
    assert np.allclose(uT_1, tf_1)
    
def test_mul():
    import pyuTensor
    import numpy as np
    import tensorflow as tf

    pyuTensor.set_ram_total(4096)
    pyuTensor.set_meta_total(4096)

    a_setting = (5, 5, 3)
    a = np.arange(np.prod(a_setting)).reshape(a_setting).astype(np.float32)
    b_setting = a_setting
    b = np.arange(np.prod(b_setting)).reshape(b_setting).astype(np.float32)

    uT_1 = pyuTensor.mul_kernel(a, b)
    np_1 = a * b
    tf_1 = tf.multiply(a, b).numpy()

    assert np.allclose(np_1, tf_1)
    assert np.allclose(uT_1, tf_1)