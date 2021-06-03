import argparse
from functools import reduce
from pathlib import Path

import numpy as np
from jinja2 import Template

test_template_str = """
/*
  Random Generated Test Number {{test_id}}
*/
TEST(StridedIterator, test_strided_it_{{test_id}}) {
  localCircularArenaAllocator<1024> meta_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  float s_input[{{input_size}}] = { {{input_arr}} };
  int32_t s_begin[{{begin_size}}] = { {{begin_arr}} };
  int32_t s_end[{{end_size}}] = { {{end_arr}} };
  int32_t s_strides[{{strides_size}}] = { {{strides_arr}} };
  int32_t ref_li[{{output_size}}] = { {{output_arr}} };

  Tensor input_tensor = new BufferTensor({ {{input_shape}} }, flt, s_input);
  Tensor begin_tensor = new BufferTensor({ {{begin_size}} }, i32, s_begin);
  Tensor end_tensor = new BufferTensor({ {{end_size}} }, i32, s_end);
  int32_t begin_mask = {{begin_mask}};
  int32_t end_mask = {{end_mask}};
  Tensor strides_tensor = new BufferTensor({ {{strides_size}} }, i32, s_strides);

  StridedIterator stride_it(input_tensor, begin_tensor, end_tensor,
                            strides_tensor, begin_mask, end_mask);
  EXPECT_EQ(stride_it.num_elems(), {{output_size}});
  // check linear index values
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
  EXPECT_EQ(stride_it.next(), -1);  // end of iteration
  // iterate over again
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
}
"""

container_template_str = """
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

{% for test_str in test_strs %}
{{ test_str }}
{% endfor %}
"""


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


def to_array_str(arr):
    return ", ".join(map(str, np.array(arr).ravel()))


def main(num_tests=5):
    test_template = Template(test_template_str)
    container_template = Template(container_template_str)
    test_strs = []
    for test_id in range(num_tests):
        shape = random_shape()
        num_elems = reduce(lambda a, b: a * b, shape, 1)
        input_arr = np.arange(num_elems).reshape(shape)
        slices = random_slices(input_arr.shape)
        out = input_arr[slices]
        begin_arr = np.array([s.start if s.start is not None else 0 for s in slices])
        end_arr = np.array(
            [s.stop if s.stop is not None else size for s, size in zip(slices, shape)]
        )
        strides_arr = np.array([s.step if s.step is not None else 1 for s in slices])
        begin_mask = get_mask(slices, "start")
        end_mask = get_mask(slices, "stop")
        test_strs.append(
            test_template.render(
                test_id=f"{test_id:02d}",
                input_size=input_arr.size,
                input_arr=to_array_str(input_arr),
                input_shape=to_array_str(input_arr.shape),
                begin_size=begin_arr.size,
                begin_arr=to_array_str(begin_arr),
                end_size=end_arr.size,
                end_arr=to_array_str(end_arr),
                strides_size=strides_arr.size,
                strides_arr=to_array_str(strides_arr),
                output_size=out.size,
                output_arr=to_array_str(out),
                begin_mask=begin_mask,
                end_mask=end_mask,
            )
        )
    test_path = (
        Path(__file__).absolute().parent.parent.parent
        / "TESTS"
        / "operators"
        / "test_stride_iterator.cpp"
    )
    with test_path.open("w") as fid:
        fid.write(container_template.render(test_strs=test_strs))
    print(f"writing testing code for strided iterator to {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tests",
        default=5,
        type=int,
        help="number of test cases (default: %(default)s)",
    )

    kwargs = vars(parser.parse_args())
    main(**kwargs)
