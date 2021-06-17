"""
NOTE: uTensor does not support broadcasting. Broadcasting is not included in the tests
"""
import argparse
import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2
from jinja_env.utensor_macros import UTENSOR_MAX_NDIMS


def random_shapes():
    shape1 = np.random.randint(1, 26, size=np.random.randint(3, UTENSOR_MAX_NDIMS + 1))
    ndims2 = np.random.randint(1, shape1.size)
    shape2 = shape1[-ndims2:]
    if shape2.size > 1:
        num_idxs = np.random.randint(1, shape2.size)
        idxs = np.random.randint(0, shape2.size, size=num_idxs)
        shape2[idxs] = 1
    return tuple(shape1), tuple(shape2)


def gen_div_test(test_number):
    test_name = f"random_gen_div__{test_number:02d}"
    shape1, shape2 = random_shapes()
    tf_in1 = tf.random.uniform(shape1, dtype=tf.float32)
    tf_in2 = tf.random.uniform(shape2, dtype=tf.float32)
    tf_out = tf.truediv(tf_in1, tf_in2)

    in1_ref_name = f"s_ref_div_in1_{test_number:02d}"
    in2_ref_name = f"s_ref_div_in2_{test_number:02d}"
    out_ref_name = f"s_ref_div_out_{test_number:02d}"
    in1_tensor = Tensor("in1", tf_in1.numpy(), ref_name=in1_ref_name)
    in2_tensor = Tensor("in2", tf_in2.numpy(), ref_name=in2_ref_name)
    ref_out_tensor = Tensor("ref_out", tf_out.numpy(), ref_name=out_ref_name)
    out_tensor = Tensor("out", tf_out.numpy())
    op = Operator("DivOperator", "div_op", dtypes=[in1_tensor.get_dtype])
    (
        op.set_inputs(
            {
                "a": in1_tensor,
                "b": in2_tensor,
            }
        )
        .set_outputs({"c": out_tensor})
        .set_namespace("ReferenceOperators::")
    )
    test = SingleOpTest("ReferenceDiv", test_name, op)
    test.add_tensor_comparison(out_tensor, ref_out_tensor, 1e-6)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def gen_mul_test(test_number):
    test_name = f"random_gen_mul__{test_number:02d}"
    shape1, shape2 = random_shapes()
    tf_in1 = tf.random.uniform(shape1, dtype=tf.float32)
    tf_in2 = tf.random.uniform(shape2, dtype=tf.float32)
    tf_out = tf.multiply(tf_in1, tf_in2)

    in1_ref_name = f"s_ref_mul_in1_{test_number:02d}"
    in2_ref_name = f"s_ref_mul_in2_{test_number:02d}"
    out_ref_name = f"s_ref_mul_out_{test_number:02d}"
    in1_tensor = Tensor("in1", tf_in1.numpy(), ref_name=in1_ref_name)
    in2_tensor = Tensor("in2", tf_in2.numpy(), ref_name=in2_ref_name)
    ref_out_tensor = Tensor("ref_out", tf_out.numpy(), ref_name=out_ref_name)
    out_tensor = Tensor("out", tf_out.numpy())
    op = Operator("MulOperator", "mul_op", dtypes=[in1_tensor.get_dtype])
    (
        op.set_inputs(
            {
                "a": in1_tensor,
                "b": in2_tensor,
            }
        )
        .set_outputs({"c": out_tensor})
        .set_namespace("ReferenceOperators::")
    )
    test = SingleOpTest("ReferenceMul", test_name, op)
    test.add_tensor_comparison(out_tensor, ref_out_tensor, 1e-6)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def gen_add_test(test_number):
    test_name = f"random_gen_add__{test_number:02d}"
    shape1, shape2 = random_shapes()
    tf_in1 = tf.random.uniform(shape1, dtype=tf.float32)
    tf_in2 = tf.random.uniform(shape2, dtype=tf.float32)
    tf_out = tf.add(tf_in1, tf_in2)

    in1_ref_name = f"s_ref_add_in1_{test_number:02d}"
    in2_ref_name = f"s_ref_add_in2_{test_number:02d}"
    out_ref_name = f"s_ref_add_out_{test_number:02d}"
    in1_tensor = Tensor("in1", tf_in1.numpy(), ref_name=in1_ref_name)
    in2_tensor = Tensor("in2", tf_in2.numpy(), ref_name=in2_ref_name)
    ref_out_tensor = Tensor("ref_out", tf_out.numpy(), ref_name=out_ref_name)
    out_tensor = Tensor("out", tf_out.numpy())
    op = Operator("AddOperator", "add_op", dtypes=[in1_tensor.get_dtype])
    (
        op.set_inputs(
            {
                "a": in1_tensor,
                "b": in2_tensor,
            }
        )
        .set_outputs({"c": out_tensor})
        .set_namespace("ReferenceOperators::")
    )
    test = SingleOpTest("ReferenceAdd", test_name, op)
    test.add_tensor_comparison(out_tensor, ref_out_tensor, 1e-6)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def gen_sub_test(test_number):
    test_name = f"random_gen_sub__{test_number:02d}"
    shape1, shape2 = random_shapes()
    tf_in1 = tf.random.uniform(shape1, dtype=tf.float32)
    tf_in2 = tf.random.uniform(shape2, dtype=tf.float32)
    tf_out = tf.subtract(tf_in1, tf_in2)

    in1_ref_name = f"s_ref_sub_in1_{test_number:02d}"
    in2_ref_name = f"s_ref_sub_in2_{test_number:02d}"
    out_ref_name = f"s_ref_sub_out_{test_number:02d}"
    in1_tensor = Tensor("in1", tf_in1.numpy(), ref_name=in1_ref_name)
    in2_tensor = Tensor("in2", tf_in2.numpy(), ref_name=in2_ref_name)
    ref_out_tensor = Tensor("ref_out", tf_out.numpy(), ref_name=out_ref_name)
    out_tensor = Tensor("out", tf_out.numpy())
    op = Operator("SubOperator", "sub_op", dtypes=[in1_tensor.get_dtype])
    (
        op.set_inputs(
            {
                "a": in1_tensor,
                "b": in2_tensor,
            }
        )
        .set_outputs({"c": out_tensor})
        .set_namespace("ReferenceOperators::")
    )
    test = SingleOpTest("ReferenceSub", test_name, op)
    test.add_tensor_comparison(out_tensor, ref_out_tensor, 1e-6)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def gen_arithmatic_broadcast(num_tests=10):
    tests = []
    const_snippets = []
    test_dir_path = (
        Path(__file__).resolve().parent.parent.parent / "TESTS" / "operators"
    )
    generators = [
        gen_sub_test,
        gen_add_test,
        gen_mul_test,
        gen_div_test,
    ]
    for test_number, generator in product(range(num_tests), generators):
        tr, cs = generator(test_number)
        tests.append(tr)
        const_snippets.extend(cs)
    with (test_dir_path / "constants_arithmetic_broadcast.hpp").open("w") as fp:
        c_r = env2.get_template("const_container.hpp").render(
            constants=const_snippets, constants_header="TEST_ARITHMATIC_H"
        )
        fp.write(c_r)
        header_fname = os.path.basename(fp.name)
        print(f"{fp.name} saved")
    with (test_dir_path / "test_arithmetic_broadcast.cpp").open("w") as fp:
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
        type=int,
        help="the number of tests for each arithmatic op (default: %(default)s)",
    )
    kwargs = vars(parser.parse_args())
    gen_arithmatic_broadcast(**kwargs)
