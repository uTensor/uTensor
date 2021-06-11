import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2


def gen_test(test_number):
    test_name = f"random_gen_fc__{test_number:02d}"
    hidden = np.random.randint(128, 513)
    if np.random.rand() > 0.5:
        bias_init = None
    else:
        bias_init = "glorot_uniform"
    layer = tf.keras.layers.Dense(hidden, use_bias=True, bias_initializer=bias_init)
    in_size = np.random.randint(128, 513)
    layer.build((1, in_size))
    tf_in = tf.random.uniform([1, in_size], dtype=tf.float32)
    tf_out = layer(tf_in)

    in_ref_name = f"s_ref_in_{test_number:02d}"
    w_ref_name = f"s_ref_w_{test_number:02d}"
    b_ref_name = f"s_ref_b_{test_number:02d}"
    out_ref_name = f"s_ref_out_{test_number:02d}"
    in_t = Tensor("in", tf_in.numpy(), ref_name=in_ref_name)
    w_t = Tensor("w", layer.kernel.numpy(), ref_name=w_ref_name)
    b_t = Tensor("b ", layer.bias.numpy(), ref_name=b_ref_name)
    out_ref = Tensor(
        "out_ref", tf_out.numpy(), ref_name=out_ref_name
    )  # Store the reference out values
    out_t = Tensor("out", tf_out.numpy())
    # conv_param_str = "{%s}, %s" % (str(strides).lstrip('[').rstrip(']'), padding)
    # convOp = Operator("Conv2dOperator", "op_0", dtypes=["float"], param_str=conv_param_str)
    param_str = "Fuseable::NoActivation<float>"
    op = Operator(
        "FullyConnectedOperator", "fcOp", dtypes=[in_t.get_dtype], param_str=param_str
    )
    op.set_inputs({"input": in_t, "filter": w_t, "bias": b_t}).set_outputs(
        {"output": out_t}
    )

    test = SingleOpTest("ReferenceFloatFullyConnect", test_name, op)
    test.add_tensor_comparison(out_t, out_ref, 1e-5)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def gen_fc_tests(num_tests):
    tests = []
    const_snippets = []
    test_dir_path = (
        Path(__file__).resolve().parent.parent.parent / "TESTS" / "operators"
    )

    for i in range(num_tests):
        tr, cs = gen_test(i)
        tests.append(tr)
        const_snippets.extend(cs)
    with (test_dir_path / "constants_float_fully_connected.hpp").open("w") as fp:
        c_r = env2.get_template("const_container.hpp").render(
            constants=const_snippets, constants_header="TEST_FLOAT_FULLYCONNECT_H"
        )
        fp.write(c_r)
        header_fname = os.path.basename(fp.name)
        print(f"{fp.name} saved")
    with (test_dir_path / "test_float_fully_connected.cpp").open("w") as fp:
        gt_r = env2.get_template("gtest_container.cpp").render(
            constants_header=header_fname,
            using_directives=["using namespace uTensor::ReferenceOperators"],
            tests=tests,
        )
        fp.write(gt_r)
        print(f"{fp.name} saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tests",
        type=int,
        help="number of tests to generate (default: %(default)s)",
        default=10,
    )
    kwargs = vars(parser.parse_args())
    gen_fc_tests(**kwargs)
