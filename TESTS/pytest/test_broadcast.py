from typing import List

def test_broadcast_shapes(random_broadcastable_shapes: List[tuple]):
    import pyuTensor
    import numpy as np

    for shape_a, shape_b in random_broadcastable_shapes:
        bc = pyuTensor.Broadcaster(shape_a, shape_b)
        uT_1 = bc.get_output_shape()
        np_1 = np.broadcast_shapes(shape_a, shape_b)
        assert uT_1 == np_1