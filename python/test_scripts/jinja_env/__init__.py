import jinja2
from pathlib import Path
import numpy as np

_template_dir = Path(__file__).parent / "templates"
_template2_dir = Path(__file__).parent / "templates_v2"

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_template_dir), trim_blocks=True, lstrip_blocks=True
    )
env.globals.update(
    zip=zip,
    len=len,
    TENSOR_TYPE_MAP={
      "int8_t": "i8",
      "uint8_t": "u8",
      "int16_t": "i16",
      "uint16_t": "u16",
      "int32_t": "i32",
      "uint32_t": "u32",
      "float": "flt",
      },
    )

del _template_dir

TENSOR_TYPE_MAP={
    "int8_t": "i8",
    "uint8_t": "u8",
    "int16_t": "i16",
    "uint16_t": "u16",
    "int32_t": "i32",
    "uint32_t": "u32",
    "float": "flt",
    }
NUMPY_2_CMAP={
    np.int8: "int8_t",
    np.uint8: "uint8_t",
    np.int16: "int16_t",
    np.uint16: "uint16_t",
    np.int32: "int32_t",
    np.uint32: "uint32_t",
    np.float: "float",
    np.dtype('float32'): "float",
    }
env2 = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_template2_dir), trim_blocks=True, lstrip_blocks=True
    )
env2.globals.update(
    zip=zip,
    len=len,
    TENSOR_TYPE_MAP=TENSOR_TYPE_MAP,
    NUMPY_2_CMAP=NUMPY_2_CMAP,
    )


class Tensor:
  def __init__(self, name, np_array, ref_name=None, quantize_params=[]):
    self.name = name
    self.np_array = np_array
    self.ref_name = ref_name
    self.quantize_params = quantize_params

  @property
  def shape(self):
    return self.np_array.shape

  @property
  def dtype(self):
    return NUMPY_2_CMAP[self.np_array.dtype]

  @property
  def utype(self):
    return TENSOR_TYPE_MAP[self.dtype]

  def flatten(self):
    return self.np_array.flatten()

  def render_constant(self):
    if self.ref_name:
      return env2.get_template('def_constant.hpp').render(tensor=self)
    else:
      return ""
  def render_declaration(self):
    if self.ref_name:
      return env2.get_template('declare_rom_tensor.cpp').render(tensor=self)
    else:
      return env2.get_template('declare_ram_tensor.cpp').render(tensor=self)


class Operator:
  def __init__(self, op_type, name, dtypes=[], param_str=None):
    self.op_type = op_type
    self.name = name
    self.dtypes = dtypes
    self.param_str = param_str
    self.array_template = env2.get_template('array_template.cpp')
    self.input_map = {}
    self.output_map = {}
    self.type_signature = env2.get_template('op_type_signature.cpp').render(op=self)

  def set_inputs(self, input_map):
    self.input_map = input_map
    return self

  def set_outputs(self, output_map):
    self.output_map = output_map
    return self

  def render_declaration(self):
    return env2.get_template('declare_operator.cpp').render(op=self)

  def render_eval(self):
    return env2.get_template('eval_operator.cpp').render(op=self)

class SingleOpTest:
  def __init__(self, test_group, test_name, target_op):
    self.test_group = test_group
    self.test_name = test_name
    self.out_size = 0
    for out_tensor in target_op.output_map:
      self.out_size += len(target_op.output_map[out_tensor].flatten())
    self.target_op = target_op
    self.compare_tensors = []
    self.tensor_set = set()
    for tensor in target_op.input_map:
      self.tensor_set.add(target_op.input_map[tensor])
    for tensor in target_op.output_map:
      self.tensor_set.add(target_op.output_map[tensor])
  
  def add_tensor_comparison(self, a, b):
    self.compare_tensors.append((a,b))
    self.tensor_set.add(a)
    self.tensor_set.add(b)

  def render(self):
    const_snippets = []
    tensor_decls = []
    for tensor in self.tensor_set:
      const_snippets.append(tensor.render_constant())
      tensor_decls.append(tensor.render_declaration())
    op_decl = self.target_op.render_declaration()
    op_eval = self.target_op.render_eval()

    compare_snippets = []
    for a, b in self.compare_tensors:
      compare_snippets.append(env2.get_template('compare_outputs.cpp').render(a=a, b=b))

    TestTemplate = env2.get_template('test_container.cpp')
    test_rendered = TestTemplate.render(test_group= self.test_group, 
                                        test_name = self.test_name,
                                        out_size  = self.out_size,
                                        tensor_declarations = tensor_decls,
                                        op_decl = op_decl,
                                        op_eval = op_eval,
                                        compare_snippets=compare_snippets)
    return (test_rendered, const_snippets)


del _template2_dir
