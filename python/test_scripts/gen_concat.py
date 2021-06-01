import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2


def _random_shapes():
    num_dims = np.random.randint(1, 5, size=1)
    cat_axis = np.random.randint(-num_dims, num_dims, size=1)[0]
    shape_a = np.random.randint(1, 15, size=num_dims)
    shape_b = shape_a.copy()
    shape_b[cat_axis] = np.random.randint(1, 15, size=1)[0]
    return tuple(shape_a), tuple(shape_b), cat_axis


def gen_test(test_group, test_id):
    test_name = f"random_gen_concat__{test_id:02d}"
    shape_a, shape_b, cat_axis = _random_shapes()
    tf_a = tf.constant(np.random.rand(*shape_a), dtype=tf.float32)
    tf_b = tf.constant(np.random.rand(*shape_b), dtype=tf.float32)
    tf_out = tf.concat([tf_a, tf_b], axis=cat_axis)

    tensor_a = Tensor("a", tf_a.numpy(), ref_name=f"ref_in_a_{test_id:02d}")
    tensor_b = Tensor("b", tf_b.numpy(), ref_name=f"ref_in_b_{test_id:02d}")
    tensor_axis = Tensor(
        "axis", np.array([cat_axis], dtype=np.int32), ref_name=f"ref_axis_{test_id:02d}"
    )
    ref_out_tensor = Tensor("ref_out", tf_out.numpy(), f"ref_out_{test_id:02d}")
    out_tensor = Tensor("out", tf_out.numpy())

    op = Operator("ConcatOperator", name="concat_op")
    op.set_namespace("ReferenceOperators::")
    op.set_inputs({"a": tensor_a, "b": tensor_b, "axis": tensor_axis}).set_outputs(
        {"out": out_tensor}
    )

    test = SingleOpTest(test_group, test_name, op)
    test.add_tensor_comparison(out_tensor, ref_out_tensor, 1e-7)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def main(test_group="ReferenceConcat", num_tests=5):
    tests = []
    const_snippets = []
    for i in range(num_tests):
        tr, const_snps = gen_test(test_group, i)
        tests.append(tr)
        const_snippets.extend(const_snps)
    output_file = str(Path("../../TESTS/operators/test_concat.cpp").resolve())
    const_file = str(Path("../../TESTS/operators/constants_concat.hpp").resolve())
    with open(const_file, "w") as fid:
        fid.write(
            env2.get_template("const_container.hpp").render(
                constants=const_snippets, constants_header="TEST_CONST_CONCAT_H"
            )
        )
    with open(output_file, "w") as fid:
        fid.write(
            env2.get_template("gtest_container.cpp").render(
                constants_header=const_file, using_directives=[], tests=tests
            )
        )
    print(f"test files generated: {output_file}, {const_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-group",
        default="ReferenceConcat",
        help="the test group name (default: %(default)s)",
    )
    parser.add_argument(
        "--num-tests",
        default=5,
        help="number of tests to generate (default: %(default)s)",
        type=int,
    )
    kwargs = vars(parser.parse_args())
    main(**kwargs)
