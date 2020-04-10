#import tensorflow as tf
import jinja2
import os
import pickle
from collections import namedtuple
import re
import numpy as np

TFLM_Tensor = namedtuple('TFLM_Tensor', ['tensor', 'quantization'])

test_name = "3_fully_connected"
output_file = "test_quantized_fully_connected.cpp"
const_file = "constants_quantized_fully_connected.hpp"

def import_test_data(dir_path="."):
    test = {}
    with open(dir_path + '/inputs.pkl','rb') as pickle_in:
        test["inputs"] = pickle.load(pickle_in)
    with open(dir_path + '/outputs.pkl','rb') as pickle_in:
        test["outputs"] = pickle.load(pickle_in)
    with open(dir_path + '/option.pkl','rb') as pickle_in:
        test["option"] = pickle.load(pickle_in)
    return test

def get_name_map(dir_path="."):
    name_map = { "inputs":{}, "outputs": {} }
    with open(dir_path + "/name_map.mp") as fp:
        for line in fp:
            line = line.lstrip()
            if line[0] == "#":
                continue
            m = re.match("(?P<mkey>\w+):\s+(?P<from>[a-zA-Z0-9_/]+)\s+->\s+(?P<to>\w+)", line)
            if m:
                g = m.groupdict()
                name_map[g["mkey"]][g["from"]] =  g["to"]
    return name_map

def dtype_to_ctype(x):
    if x == "int8":
        return "int8_t"
    elif x == "uint8":
        return "uint8_t"
    elif x == "int16":
        return "int16_t"
    elif x == "uint16":
        return "uint16_t"
    elif x == "int32":
        return "int32_t"
    elif x == "uint32":
        return "uint32_t"
    else:
        print("unexpected DTYPE", x)
        return None

def dtype_to_utype(x):
    if x == "int8":
        return "i8"
    elif x == "uint8":
        return "u8"
    elif x == "int16":
        return "i16"
    elif x == "uint16":
        return "u16"
    elif x == "int32":
        return "i32"
    elif x == "uint32":
        return "u32"
    elif x == "float":
        return "flt"
    else:
        print("unexpected DTYPE", x)
        return None

const_str = """
{% for tensor in reference_tensors %} 
static const {{ tensor.type }} {{ tensor.r_name }}[{{ tensor.flat_num_elems }}] = { 
{% for x in tensor.data  %} {{ x }}{{ "," if not loop.last }}{{ "\n" if loop.index0 != 0 and loop.index0 % 10 == 0 }} {% endfor %} 
};
static const int32_t {{ tensor.r_zp_name }} [1] = { {{ tensor.zp }} };
static const float {{ tensor.r_scale_name }} [1] = { {{ tensor.scale }} };

{% endfor %}
"""

test_str = """

TEST(Quantized, reference_{{ test_name }}) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<{{ out_tensor.flat_num_elems }}*2*sizeof({{ out_tensor.utype }}), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  {% for tensor in reference_tensors %}
  Tensor {{ tensor.name }} =  new RomTensor({ {% for x in tensor.shape %}{{ x }}{{ ", " if not loop.last }}{% endfor %} }, {{ tensor.utype }}, {{ tensor.r_name }})
                        .set_quantization_params(PerTensorQuantizationParams({{ tensor.r_zp_name }}, {{ tensor.r_scale_name }}));

  {% endfor %}
  Tensor out = new RamTensor({ {% for x in out_tensor.shape %}{{ x }}{{ ", " if not loop.last }}{% endfor %} }, {{ out_tensor.utype }})
                .set_quantization_params(PerTensorQuantizationParams({{ out_tensor.r_zp_name }}, {{ out_tensor.r_scale_name }} ));

  /*
  ConvOperator<float> conv_Aw({ {% for x in strides %}{{ x }}{{ "," if not loop.last }}{% endfor %}}, {{ PADDING }});
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();
  */
  for(int i = 0; i < {{ out_size }}; i++) {
    //EXPECT_NEAR((float) out(i), {{ out_tensor.r_name }}[i], 0.0001);
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

{{ test }}

"""
const_container_str = """
#ifndef {{ constants_header | replace(".", "_") }} 
#define {{ constants_header | replace(".", "_") }} 
{{ constant_snippet }}
#endif
"""

test_Template = jinja2.Template(test_str)
const_Template = jinja2.Template(const_str)
container_Template = jinja2.Template(container_str)
const_container_Template = jinja2.Template(const_container_str)


if __name__ == "__main__":
    tests=[]
    constants=[]
    reference_tensors = []
    out_tensor = {}
    
    ref_test_data = import_test_data()
    name_map = get_name_map()

    for key in name_map:
        for t in ref_test_data[key]:
            tensor = {}
            test = ref_test_data[key][t]
            test_data = test.tensor
            (q_scale, q_zp) = test.quantization
            tensor["name"] = name_map[key][t]
            tensor["r_name"] = "s_ref_%s" % tensor["name"]
            tensor["type"] = dtype_to_ctype(test_data.dtype)
            tensor["utype"] = dtype_to_utype(test_data.dtype)
            tensor["shape"]  = test_data.shape
            tensor["data"] = test_data.flatten()
            tensor["flat_num_elems"] = len(tensor["data"])
            tensor["zp"] = q_zp
            tensor["scale"] = q_scale
            tensor["r_zp_name"] = tensor["r_name"] + "_zp"
            tensor["r_scale_name"] = tensor["r_name"] + "_scale"
            tensor["io"] = key

            # hack for now
            if key == "outputs":
                out_tensor = tensor
            reference_tensors.append(tensor)

    #print(reference_tensors)
    const_str_rendered = const_Template.render(reference_tensors=reference_tensors)
    const_container_rendered = const_container_Template.render(constant_snippet=const_str_rendered, constants_header=const_file)
    test_str_rendered = test_Template.render(test_name=test_name, reference_tensors=reference_tensors, out_tensor=out_tensor)
    container_rendered = container_Template.render(test=test_str_rendered, constants_header=const_file)
    with open(const_file, "w") as fp:
        fp.write(const_container_rendered)
    with open(output_file, "w") as fp:
        fp.write(container_rendered)


#for i in range(num_tests):
#    for stride in [1, 2]:
#        w =  tf.Variable(tf.random.normal([5, 5, 1, 32]))
#        in_1 = tf.Variable(tf.random.normal([1, 28, 28, 1]))
#        strides = [1, stride, stride, 1]
#        out_1 = tf.nn.conv2d(in_1, w, strides=strides, padding=PADDING)
#        w_flat = w.numpy().flatten()
#        in_flat = in_1.numpy().flatten()
#        out_flat = out_1.numpy().flatten()
#        stride_str = "_stride_%d" % stride
#        test_name = "%s_%d%s" % (PADDING, i, stride_str)
#
#        test_str_rendered = test_Template.render(test_name=test_name, input_size=in_flat.shape[0], w_size=w_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_w=w_flat, ref_out=out_flat, strides=strides, in_shape=in_1.shape, w_shape=w.shape, out_shape=out_1.shape, PADDING=PADDING)
#        const_str_rendered = const_Template.render(test_name=test_name, input_size=in_flat.shape[0], w_size=w_flat.shape[0], out_size=out_flat.shape[0], ref_in=in_flat, ref_w=w_flat, ref_out=out_flat, strides=strides, in_shape=in_1.shape, w_shape=w.shape, out_shape=out_1.shape)
#        tests.append(test_str_rendered)
#        constants.append(const_str_rendered)
#
#container_rendered = container_Template.render(tests=tests, constants_header=const_file)
#consts_container_rendered = const_container_Template.render(constants=constants, constants_header=const_file)
#with open(output_file, "w") as fp:
#    fp.write(container_rendered)
#with open(const_file, "w") as fp:
#    fp.write(consts_container_rendered)

print("Complete");
