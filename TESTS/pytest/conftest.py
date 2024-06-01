import pytest
from random import randint, random, choice

@pytest.fixture(scope="function")
def random_broadcastable_shapes():
    pairs = []
    # 1D
    for _ in range(10):
        shape_a = (randint(1, 10),)
        shape_b = choice([(1,), shape_a, (1,)+shape_a])
        pairs.append((shape_a, shape_b))
    # ND
    for _ in range(20):
        shape_a = tuple(randint(1, 10) for _ in range(randint(1, 4)))
        shape_b = tuple(d if random()>0.5 else 1 for d in shape_a)
        pairs.append((shape_a, shape_b))
        if len(shape_a) < 4:
            shape_b = tuple(1 for _ in range(randint(1, 4-len(shape_a)))) + shape_a
            pairs.append((shape_a, shape_b))
        else:
            pairs.append((shape_a, shape_a[-randint(1, 3):]))
    
    return pairs