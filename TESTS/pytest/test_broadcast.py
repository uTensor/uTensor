from typing import List

def test_broadcast_shapes(random_broadcastable_shapes: List[tuple]):
    import pyuTensor
    import numpy as np

    pyuTensor.set_ram_total(100*1024)
    pyuTensor.set_meta_total(100*1024)

    for shape_a, shape_b in random_broadcastable_shapes:
        a = np.arange(np.prod(shape_a)).reshape(shape_a).astype(np.float32)
        b = np.arange(np.prod(shape_b)).reshape(shape_b).astype(np.float32)

        bc = pyuTensor.Broadcaster(shape_a, shape_b)
        uT_1 = bc.get_shape_c()
        np_1 = np.broadcast_shapes(shape_a, shape_b)
        assert isinstance(uT_1, tuple)
        assert uT_1 == np_1