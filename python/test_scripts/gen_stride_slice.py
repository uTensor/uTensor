import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from jinja_env import Operator, SingleOpTest, Tensor, env2


def random_slices(shape):
    slices = []
    for size in shape:
        if np.random.rand() > 0.5:
            start = None
        else:
            start = np.random.randint(0, size)
        if np.random.rand() > 0.5:
            stop = None
        else:
            if start is None:
                stop = np.random.randint(1, size + 1)
            else:
                stop = np.random.randint(start + 1, size + 1)
        if np.random.rand() > 0.5:
            step = None
        else:
            step = np.random.randint(1, 4)
        slices.append(slice(start, stop, step))
    return tuple(slices)


def random_shape():
    num_dims = np.random.randint(1, 5)
    return tuple(np.random.randint(1, 10) for _ in range(num_dims))


def get_mask(slices, name):
    mask = 0
    for i, sl in enumerate(slices):
        if getattr(sl, name) is None:
            mask |= 1 << i
    return mask


def gen_test(test_id):
    test_name = f"random_gen_strided_slice__{test_id:02d}"
    in_shape = random_shape()
    slices = random_slices(in_shape)
    tf_x = tf.constant(np.random.rand(*in_shape), dtype=tf.float32)
    tf_out = tf_x[slices]

    begin_mask = get_mask(slices, "start")
    end_mask = get_mask(slices, "stop")
    tensor_x = Tensor("x", tf_x.numpy(), ref_name=f"ref_in_x_{test_id:02d}")
    tensor_begin = Tensor(
        "tensor_begin",
        np.array(
            [s.start if s.start is not None else 0 for s in slices], dtype=np.int32
        ),
        ref_name=f"ref_begin_{test_id:02d}",
    )
    tensor_end = Tensor(
        "tensor_end",
        np.array(
            [
                s.stop if s.stop is not None else size
                for s, size in zip(slices, in_shape)
            ],
            dtype=np.int32,
        ),
        ref_name=f"ref_end_{test_id:02d}",
    )
    tensor_strides = Tensor(
        "tensor_strides",
        np.array([s.step if s.step is not None else 1 for s in slices], dtype=np.int32),
        ref_name=f"ref_strides_{test_id:02d}",
    )
    ref_out_tensor = Tensor(
        "ref_tensor_output", tf_out.numpy(), f"ref_out_{test_id:02d}"
    )
    out_tensor = Tensor("tensor_output", tf_out.numpy())

    op = Operator(
        "StridedSliceOperator",
        name="strided_slice_op",
        dtypes=[tensor_x.get_dtype],
        param_str=f"{begin_mask}, {end_mask}, 0, 0, 0",
    )
    op.set_namespace("ReferenceOperators::")
    op.set_inputs(
        {
            "input": tensor_x,
            "begin": tensor_begin,
            "end": tensor_end,
            "strides": tensor_strides,
        }
    ).set_outputs({"output": out_tensor})

    test = SingleOpTest("ReferenceStridedSlice", test_name, op)
    test.add_tensor_comparison(out_tensor, ref_out_tensor, 1e-7)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets


def main(num_tests=5):
    tests = []
    const_snippets = []
    for i in range(num_tests):
        tr, const_snps = gen_test(i)
        tests.append(tr)
        const_snippets.extend(const_snps)
    output_path = Path("../../TESTS/operators/test_stride_slice.cpp").resolve()
    const_path = Path("../../TESTS/operators/constants_stride_slice.hpp").resolve()
    with const_path.open("w") as fid:
        fid.write(
            env2.get_template("const_container.hpp").render(
                constants=const_snippets, constants_header="TEST_CONST_CONCAT_H"
            )
        )
    with output_path.open("w") as fid:
        fid.write(
            env2.get_template("gtest_container.cpp").render(
                constants_header=const_path.name, using_directives=[], tests=tests
            )
        )
    print(f"test files generated: {output_path}, {const_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tests",
        default=10,
        help="number of tests to generate (default: %(default)s)",
        type=int,
    )
    kwargs = vars(parser.parse_args())
    main(**kwargs)
