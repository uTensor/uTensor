import os
from random import choice, randint, random

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@pytest.fixture(scope="function")
def random_broadcastable_shapes():
    pairs = []
    # 1D
    for _ in range(10):
        shape_a = (randint(1, 10),)
        shape_b = choice([(1,), shape_a, (1,) + shape_a])
        pairs.append((shape_a, shape_b))
    # ND
    for _ in range(20):
        shape_a = tuple(randint(1, 10) for _ in range(randint(1, 4)))
        shape_b = tuple(d if random() > 0.5 else 1 for d in shape_a)
        pairs.append((shape_a, shape_b))
        if len(shape_a) < 4:
            shape_b = tuple(1 for _ in range(randint(1, 4 - len(shape_a)))) + shape_a
            pairs.append((shape_a, shape_b))
        else:
            pairs.append((shape_a, shape_a[-randint(1, 3) :]))

    return pairs


@pytest.fixture(scope="function")
def random_conv_input():
    import numpy as np

    inputs = []
    rand_max = 10

    for _ in range(100):
        B, H, W, Cin = (
            randint(1, rand_max),
            randint(1, rand_max),
            randint(1, rand_max),
            randint(1, rand_max),
        )
        kernelH, kernelW, Cout = randint(1, H), randint(1, W), randint(1, rand_max)
        bias = np.random.rand(Cout).astype(np.float32)
        strides = [1, randint(1, 3), randint(1, 3), 1]
        padding = choice(["VALID", "SAME"])
        input_tensor = np.random.rand(B, H, W, Cin).astype(np.float32)
        filter_tensor = np.random.rand(kernelH, kernelW, Cin, Cout).astype(np.float32)
        inputs.append((input_tensor, filter_tensor, bias, strides, padding))

    return inputs

@pytest.fixture(scope="function")
def random_4d_input():
    import numpy as np

    inputs = []
    rand_max = 10

    for _ in range(100):
        B, H, W, Cin = (
            randint(1, rand_max),
            randint(1, rand_max),
            randint(1, rand_max),
            randint(1, rand_max),
        )
        input_tensor = np.random.rand(B, H, W, Cin).astype(np.float32)
        inputs.append(input_tensor)

    return inputs