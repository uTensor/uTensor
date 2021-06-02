import numpy as np
import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2

test_group = "ReferenceFC"
num_tests = 5
output_file = "test_fully_connected.cpp"
const_file = "constants_fully_connected.hpp"


def gen_test(test_number):
    test_name = "random_gen_fc__%d" % (test_number)
    in0 = tf.constant(tf.random.uniform([1, 2, 2, 64])).numpy()
    w = tf.constant(tf.random.uniform([512, 256])).numpy()
    if test_number == 0:
        b = np.zeros([1, 512], dtype=np.float32).flatten()
    else:
        b = tf.constant(tf.random.uniform([1, 512])).numpy().flatten()
    # Combine ops to behave like final kernel
    in1 = tf.reshape(in0, (1, -1)).numpy()
    m = tf.linalg.matmul(in1, w, transpose_b=True)
    print(m.shape)
    out_1 = tf.math.add(m, b).numpy()
    w = np.transpose(w)

    in_ref_name = "s_ref_in_%d" % test_number
    w_ref_name = "s_ref_w_%d" % test_number
    b_ref_name = "s_ref_b_%d" % test_number
    out_ref_name = "s_ref_out_%d" % test_number
    in_t = Tensor("in", in1, ref_name=in_ref_name)
    w_t = Tensor("w", w, ref_name=w_ref_name)
    b_t = Tensor("b ", b, ref_name=b_ref_name)
    out_ref = Tensor(
        "out_ref", out_1, ref_name=out_ref_name
    )  # Store the reference out values
    out_t = Tensor("out", out_1)
    # conv_param_str = "{%s}, %s" % (str(strides).lstrip('[').rstrip(']'), padding)
    # convOp = Operator("Conv2dOperator", "op_0", dtypes=["float"], param_str=conv_param_str)
    param_str = "Fuseable::NoActivation<float>"
    op = Operator(
        "FullyConnectedOperator", "fcOp", dtypes=[in_t.get_dtype], param_str=param_str
    )
    op.set_inputs({"input": in_t, "filter": w_t, "bias": b_t}).set_outputs(
        {"output": out_t}
    )

    test = SingleOpTest(test_group, test_name, op)
    test.add_tensor_comparison(out_t, out_ref)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


if __name__ == "__main__":
    tests = []
    const_snippets = []
    for i in range(num_tests):
        tr, cs = gen_test(i)
        tests.append(tr)
        const_snippets.extend(cs)
    with open(const_file, "w") as fp:
        c_r = env2.get_template("const_container.hpp").render(
            constants=const_snippets, constants_header=const_file
        )
        fp.write(c_r)
    with open(output_file, "w") as fp:
        gt_r = env2.get_template("gtest_container.cpp").render(
            constants_header=const_file,
            using_directives=["using namespace uTensor::ReferenceOperators"],
            tests=tests,
        )
        fp.write(gt_r)
