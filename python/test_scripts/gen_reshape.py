import os
from functools import reduce
from pathlib import Path

import numpy as np
import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2


def gen_test(test_id, old_shape, new_shape):
    mul = lambda a, b: a * b
    assert reduce(mul, old_shape, 1) == reduce(mul, new_shape)
    test_name = f"random_gen_reshape__{test_id:02d}"
    tf_in = tf.random.uniform(old_shape, dtype=tf.float32)
    tf_out = tf.reshape(tf_in, new_shape)

    in_ref_name = f"s_ref_in_{test_id:02d}"
    out_ref_name = f"s_ref_out_{test_id:02d}"
    in_t = Tensor("in", tf_in.numpy(), ref_name=in_ref_name)
    out_ref = Tensor("out_ref", tf_out.numpy(), ref_name=out_ref_name)
    out_t = Tensor("out", tf_out.numpy())
    op = Operator(
        "ReshapeOperator",
        "reshape_op",
        dtypes=[in_t.get_dtype],
        param_str=f'{{{", ".join(map(str, new_shape))}}}',
    )
    (
        op.set_namespace("ReferenceOperators::")
        .set_inputs({"input": in_t})
        .set_outputs({"output": out_t})
    )

    test = SingleOpTest("ReferenceReshape", test_name, op)
    test.add_tensor_comparison(out_t, out_ref)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def gen_reshape():
    tests = []
    const_snippets = []
    test_dir_path = (
        Path(__file__).resolve().parent.parent.parent / "TESTS" / "operators"
    )
    cases = [
        ((3, 5, 4), (2, 2, 3, 5)),
        ((10, 3), (2, 5, 3)),
        ((1, 5), (1, 1, 5)),
        ((3, 5), (1, 3, 5)),
        ((3, 5), (3, 1, 5)),
        ((3, 5), (3, 5, 1)),
    ]
    for i, (old_shape, new_shape) in enumerate(cases):
        tr, cs = gen_test(i, old_shape, new_shape)
        tests.append(tr)
        const_snippets.extend(cs)
    with (test_dir_path / "constants_reshape.hpp").open("w") as fp:
        c_r = env2.get_template("const_container.hpp").render(
            constants=const_snippets, constants_header="TEST_FLOAT_TANH_H"
        )
        fp.write(c_r)
        header_fname = os.path.basename(fp.name)
        print(f"{fp.name} saved")
    with (test_dir_path / "test_reshape.cpp").open("w") as fp:
        gt_r = env2.get_template("gtest_container.cpp").render(
            constants_header=header_fname,
            tests=tests,
        )
        fp.write(gt_r)
        print(f"{fp.name} saved")


if __name__ == "__main__":
    gen_reshape()
