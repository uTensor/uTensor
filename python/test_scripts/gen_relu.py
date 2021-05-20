import tensorflow as tf
import numpy as np
import jinja2

output_file = "test_relu.cpp"
const_file = "constants_relu.hpp"


const_str = """
static const {{ dtype }} s_in_{{ test_name }}[{{ input_size }}] = { {% for x in ref_in %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
static const {{ dtype }} s_ref_out_{{ test_name }}[{{ out_size }}] = { {% for x in ref_out %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
"""

test_str = """

TEST(ReLU, random_inputs_{{ test_name }}) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<{{ out_size }}*2*sizeof({{ dtype }}), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ {% for x in in_shape %}{{ x }}{{ "," if not loop.last }}{% endfor %} }, {{ u_dtype }});
  for(int i = 0; i < {{ input_size }}; i++) {
    io(i) = s_in_{{ test_name }}[i];
  }

  InPlaceReLU<{{ dtype }}> ReLU;
  ReLU
       .set_inputs({ {InPlaceReLU<{{ dtype }}>::x, io}})
       .eval();

  {{ dtype }} tmp;
  for(int i = 0; i < {{ out_size }}; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_{{ test_name }}[i]), 0, 0.0001);
  }
}
"""

container_str = """
#include <cstring>
#include <iostream>

#include "ActivationFncs.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "uTensor/core/context.hpp"
#include "gtest/gtest.h"

#include "{{ constants_header }}"
using std::cout;
using std::endl;

using namespace uTensor;

{% for test in tests %}
/*********************************************
 * Generated Test number {{ loop.counter }}
 *********************************************/
{{ test }}

{% endfor %}
"""
const_container_str = """
#ifndef {{ constants_header | replace(".", "_") }} 
#define {{ constants_header | replace(".", "_") }} 
{% for constant_snippet in constants %}
{{ constant_snippet }}
{% endfor %}
#endif
"""

test_Template = jinja2.Template(test_str)
const_Template = jinja2.Template(const_str)
container_Template = jinja2.Template(container_str)
const_container_Template = jinja2.Template(const_container_str)

num_tests = 5
tests=[]
constants=[]
for test_type in [(np.float32, "float", "flt"), (np.int8, "int8_t", "i8"), (np.int16, "int16_t", "i16"), (np.int32, "int32_t", "i32")]:
    for i in range(num_tests):
        np_dtype, dtype, u_dtype = test_type
        in_1 = tf.Variable(tf.random.normal([1, 28, 28, 1]))
        out_1 = tf.nn.relu(in_1)
        in_flat = in_1.numpy().astype(np_dtype).flatten()
        out_flat = out_1.numpy().astype(np_dtype).flatten()
        test_name = "%s_%d" % (dtype, i)

        test_str_rendered = test_Template.render(test_name=test_name, input_size=in_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_out=out_flat, in_shape=in_1.shape, out_shape=out_1.shape, dtype=dtype, u_dtype=u_dtype)
        const_str_rendered = const_Template.render(test_name=test_name, input_size=in_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_out=out_flat, in_shape=in_1.shape, out_shape=out_1.shape, dtype=dtype)
        tests.append(test_str_rendered)
        constants.append(const_str_rendered)

container_rendered = container_Template.render(tests=tests, constants_header=const_file)
consts_container_rendered = const_container_Template.render(constants=constants, constants_header=const_file)
with open(output_file, "w") as fp:
    fp.write(container_rendered)
with open(const_file, "w") as fp:
    fp.write(consts_container_rendered)

print("Complete");
