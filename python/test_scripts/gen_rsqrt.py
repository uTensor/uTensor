from random import randint

import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2

test_group = "ReferenceRsqrt"
num_tests = 5
output_file = "test_rsqrt.cpp"
const_file = "constants_rsqrt.hpp"


def gen_test(test_number: int):
    test_name = f"random_gen_rsqrt__{test_number:d}"
    num_dims = randint(2, 4)
    in_tensor = tf.constant(
        tf.random.uniform([randint(3, 10) for _ in range(num_dims)]),
    ).numpy()
    out = tf.math.rsqrt(in_tensor).numpy()
    in_t = Tensor("input", in_tensor, f"s_ref_in_{test_number:0d}")
    out_ref = Tensor("out_ref", out, f"s_ref_out_{test_number:0d}")
    out_t = Tensor("out", out)
    op = Operator(
        "RsqrtOperator", "rsqrt_op", dtypes=[lambda: "float"]
    )
    op.set_namespace("uTensor::ReferenceOperators::")
    op.set_inputs({"input": in_t}).set_outputs({"output": out_t})
    test = SingleOpTest(test_group, test_name, op)
    test.add_tensor_comparison(out_t, out_ref, 0.001)
    return test.render()


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
