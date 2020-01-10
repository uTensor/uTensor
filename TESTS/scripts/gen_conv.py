import tensorflow as tf
import jinja2

output_file = "test_convolution.cpp"
const_file = "constants_convolution.hpp"

const_str = """
static const float s_in_{{ test_number }}[{{ input_size }}] = { {% for x in ref_in %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
static const float s_w_{{ test_number }}[{{ w_size }}] = { {% for x in ref_w %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
static const float s_ref_out_{{ test_number }}[{{ out_size }}] = { {% for x in ref_out %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
"""

test_str = """

TEST(Convolution, random_inputs_{{test_number}}) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<{{ out_size }}*2*sizeof(float)> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ {% for x in in_shape %}{{ x }}{{ "," if not loop.last }}{% endfor %} }, flt, s_in_{{ test_number }});
  Tensor w = new RomTensor({ {% for x in w_shape %}{{ x }}{{ "," if not loop.last }}{% endfor %} }, flt, s_w_{{ test_number }});
  Tensor out = new RamTensor({ {% for x in out_shape %}{{ x }}{{ "," if not loop.last }}{% endfor %} }, flt);

  ConvOperator<float> conv_Aw({ {% for x in strides %}{{ x }}{{ "," if not loop.last }}{% endfor %}}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < {{ out_size }}; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_{{ test_number }}[i], 0.0001);
  }
}
"""

container_str = """
#include <cstring>
#include <iostream>

#include "Convolution.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
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
for i in range(num_tests):
    for stride in [1, 2]:
        w =  tf.Variable(tf.random.normal([5, 5, 1, 32]))
        in_1 = tf.Variable(tf.random.normal([1, 28, 28, 1]))
        strides = [1, stride, stride, 1]
        out_1 = tf.nn.conv2d(in_1, w, strides=strides, padding='SAME')
        w_flat = w.numpy().flatten()
        in_flat = in_1.numpy().flatten()
        out_flat = out_1.numpy().flatten()
        stride_str = "_stride_%d" % stride
        test_number = "%d%s" % (i, stride_str)

        test_str_rendered = test_Template.render(test_number=test_number, input_size=in_flat.shape[0], w_size=w_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_w=w_flat, ref_out=out_flat, strides=strides, in_shape=in_1.shape, w_shape=w.shape, out_shape=out_1.shape)
        const_str_rendered = const_Template.render(test_number=test_number, input_size=in_flat.shape[0], w_size=w_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_w=w_flat, ref_out=out_flat, strides=strides, in_shape=in_1.shape, w_shape=w.shape, out_shape=out_1.shape)
        tests.append(test_str_rendered)
        constants.append(const_str_rendered)

container_rendered = container_Template.render(tests=tests, constants_header=const_file)
consts_container_rendered = const_container_Template.render(constants=constants, constants_header=const_file)
with open(output_file, "w") as fp:
    fp.write(container_rendered)
with open(const_file, "w") as fp:
    fp.write(consts_container_rendered)

print("Complete");
