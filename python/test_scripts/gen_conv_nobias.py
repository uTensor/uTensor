import tensorflow as tf
import numpy as np
import jinja2

output_file = "test_convolution_nobias.cpp"
const_file = "constants_convolution_nobias.hpp"

PADDING = "VALID"

const_str = """
static const float s_in_{{ test_name }}[{{ input_size }}] = { {% for x in ref_in %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
static const float s_w_{{ test_name }}[{{ w_size }}] = { {% for x in ref_w %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
static const float s_ref_out_{{ test_name }}[{{ out_size }}] = { {% for x in ref_out %} {{ x }}{{ "," if not loop.last }} {% endfor %} };
"""

test_str = """

TEST(ConvolutionNoBias, random_inputs_{{ test_name }}) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<{{ out_size }}*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ {% for x in in_shape %}{{ x }}{{ "," if not loop.last }}{% endfor %} }, flt, s_in_{{ test_name }});
  Tensor w = new RomTensor({ {% for x in w_shape %}{{ x }}{{ "," if not loop.last }}{% endfor %} }, flt, s_w_{{ test_name }});
  Tensor out = new RamTensor({ {% for x in out_shape %}{{ x }}{{ "," if not loop.last }}{% endfor %} }, flt);

  Conv2dOperator<float> conv_Aw({ {% for x in strides %}{{ x }}{{ "," if not loop.last }}{% endfor %}}, {{ PADDING }});
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < {{ out_size }}; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_{{ test_name }}[i], 0.0001);
  }
}
"""

container_str = """
#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "{{ constants_header }}"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

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

num_tests = 3
tests=[]
constants=[]
for padding in ["SAME", "VALID"]:
  for i in range(num_tests):
    for stride in [1, 2]:
      w0 =  tf.constant(tf.random.uniform([5, 5, 3, 32]))
      in_1 = tf.constant(tf.random.uniform([1, 28, 28, 3]))
      strides = [1, stride, stride, 1]
      out_1 = tf.nn.conv2d(in_1, w0, strides=strides, padding=padding)
      # TF2 usess a different ordering of weights
      w0_n = w0.numpy()
      w = np.zeros((w0.shape[3], w0.shape[0], w0.shape[1], w0.shape[2]))
      for i0 in range(w.shape[0]):
        for i1 in range(w.shape[1]):
          for i2 in range(w.shape[2]):
            for i3 in range(w.shape[3]):
              w[i0, i1, i2, i3] = w0_n[i1, i2, i3, i0]

      w_flat = w.flatten()
      in_flat = in_1.numpy().flatten()
      out_flat = out_1.numpy().flatten()
      stride_str = "_stride_%d" % stride
      test_name = "%s_%d%s" % (padding, i, stride_str)
  
      test_str_rendered = test_Template.render(test_name=test_name, input_size=in_flat.shape[0], w_size=w_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_w=w_flat, ref_out=out_flat, strides=strides, in_shape=in_1.shape, w_shape=w.shape, out_shape=out_1.shape, PADDING=padding)
      const_str_rendered = const_Template.render(test_name=test_name, input_size=in_flat.shape[0], w_size=w_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_w=w_flat, ref_out=out_flat, strides=strides, in_shape=in_1.shape, w_shape=w.shape, out_shape=out_1.shape)
      tests.append(test_str_rendered)
      constants.append(const_str_rendered)

container_rendered = container_Template.render(tests=tests, constants_header=const_file)
consts_container_rendered = const_container_Template.render(constants=constants, constants_header=const_file)
with open(output_file, "w") as fp:
  fp.write(container_rendered)
with open(const_file, "w") as fp:
  fp.write(consts_container_rendered)

print("Complete");
