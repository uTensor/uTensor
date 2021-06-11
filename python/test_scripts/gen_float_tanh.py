import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2


def gen_test(test_id):
    test_name = f"random_gen_float_tanh__{test_id:02d}"
    num_elems = np.random.randint(10, 51)
    tf_in = tf.random.uniform([1, num_elems], dtype=tf.float32)
    tf_out = tf.math.tanh(tf_in)

    in_ref_name = f"s_ref_in_{test_id:02d}"
    out_ref_name = f"s_ref_out_{test_id:02d}"
    in_t = Tensor("in", tf_in.numpy(), ref_name=in_ref_name)
    out_ref = Tensor("out_ref", tf_out.numpy(), ref_name=out_ref_name)
    out_t = Tensor("out", tf_out.numpy())
    op = Operator("TanhOperator", "tanh_op", dtypes=[out_t.get_dtype, in_t.get_dtype])
    (
        op.set_namespace("ReferenceOperators::")
        .set_inputs({"act_in": in_t})
        .set_outputs({"act_out": out_t})
    )

    test = SingleOpTest("ReferenceFloatTanh", test_name, op)
    test.add_tensor_comparison(out_t, out_ref, 1e-5)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def gen_float_tanh(num_tests=10):
    tests = []
    const_snippets = []
    test_dir_path = (
        Path(__file__).resolve().parent.parent.parent / "TESTS" / "operators"
    )
    for i in range(num_tests):
        tr, cs = gen_test(i)
        tests.append(tr)
        const_snippets.extend(cs)
    with (test_dir_path / "constants_float_tanh.hpp").open("w") as fp:
        c_r = env2.get_template("const_container.hpp").render(
            constants=const_snippets, constants_header="TEST_FLOAT_TANH_H"
        )
        fp.write(c_r)
        header_fname = os.path.basename(fp.name)
        print(f"{fp.name} saved")
    with (test_dir_path / "test_float_tanh.cpp").open("w") as fp:
        gt_r = env2.get_template("gtest_container.cpp").render(
            constants_header=header_fname,
            tests=tests,
        )
        fp.write(gt_r)
        print(f"{fp.name} saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tests",
        default=10,
        help="the number of test cases (default: %(default)s)",
        type=int,
    )
    kwargs = vars(parser.parse_args())
    gen_float_tanh(**kwargs)
