from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path

import jinja2
import numpy as np

from .quantization_util import get_quantization_params, quantize

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

TENSOR_TYPE_MAP = {
    "int8_t": "i8",
    "uint8_t": "u8",
    "int16_t": "i16",
    "uint16_t": "u16",
    "int32_t": "i32",
    "uint32_t": "u32",
    "float": "flt",
}
NUMPY_2_CMAP = {
    np.int8: "int8_t",
    np.uint8: "uint8_t",
    np.int16: "int16_t",
    np.uint16: "uint16_t",
    np.int32: "int32_t",
    np.uint32: "uint32_t",
    np.float: "float",
    np.dtype("int8"): "int8_t",
    np.dtype("uint8"): "uint8_t",
    np.dtype("int16"): "int16_t",
    np.dtype("uint16"): "uint16_t",
    np.dtype("int32"): "int32_t",
    np.dtype("uint32"): "uint32_t",
    np.dtype("float32"): "float",
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


class QuantizationType(Enum):
    NONE = 0
    PER_TENSOR_ASYMMETRIC = 1
    PER_CHANNEL_ASYMMETRIC = 2
    PER_TENSOR_SYMMETRIC = 3
    PER_CHANNEL_SYMMETRIC = 4


class UnknownQuantizationTypeError(Exception):
    pass


class QuantizationParams(object):
    def __init__(self, tensor):
        self.tensor = tensor  # Store ref to parent
        self.ref_name = tensor.ref_name
        self.zp = []
        self.scale = []
        # self.num_channels = 0

    @property
    def ref_zp(self):
        if not self.ref_name:
            print("WARNING: No reference name set for Quantization Param")
        return "%s_zp" % self.ref_name

    @property
    def ref_scale(self):
        if not self.ref_name:
            print("WARNING: No reference name set for Quantization Param")
        return "%s_scale" % self.ref_name

    def render_set_quantization_params(self):
        if self.zp:
            return env2.get_template("set_quantization_params.cpp").render(qp=self)
        else:
            return ""

    @property
    def num_channels(self):
        if self.zp:
            return len(self.zp)
        else:
            return 0

    @property
    def quantization_type(self):
        if self.num_channels == 1:
            return "PerTensorQuantizationParams"
        elif self.num_channels > 1:
            return "PerChannelQuantizationParams"
        else:
            raise UnknownQuantizationTypeError


class Tensor:
    def __init__(
        self,
        name,
        np_array,
        ref_name=None,
        quantization_type=QuantizationType.NONE,
        quantize_dim=None,
        narrow_range=False,
        num_quant_bits=8,
    ):
        self.name = name
        self.np_array = np_array
        self.ref_name = ref_name
        self.quantize_params = QuantizationParams(self)
        self.quantization_type = quantization_type
        self.quantize_dim = quantize_dim
        self.narrow_range = narrow_range
        self.quantized = False
        self.num_quant_bits = num_quant_bits

    @property
    def shape(self):
        return self.np_array.shape

    @property
    def dtype(self):
        return NUMPY_2_CMAP[self.np_array.dtype]

    def get_dtype(self):
        return self.dtype

    @property
    def utype(self):
        return TENSOR_TYPE_MAP[self.dtype]

    def flatten(self):
        return self.np_array.flatten()

    def render_constant(self):
        return env2.get_template("def_constant.hpp").render(tensor=self)

    def render_declaration(self):
        if self.ref_name:
            return env2.get_template("declare_rom_tensor.cpp").render(tensor=self)
        else:
            return env2.get_template("declare_ram_tensor.cpp").render(tensor=self)

    def is_quantized(self):
        return self.quantized and self.is_quantizable()

    def is_quantizable(self):
        return self.quantization_type != QuantizationType.NONE

    @property
    def symmetric(self):
        if self.is_quantizable() and (
            self.quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC
            or self.quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC
        ):
            return True
        else:
            return False

    @property
    def per_tensor_quantization(self):
        return self.is_quantizable() and (
            self.quantization_type == QuantizationType.PER_TENSOR_ASYMMETRIC
            or self.quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC
        )

    @property
    def per_channel_quantization(self):
        return (
            self.is_quantizable()
            and self.quantize_dim != None
            and (
                self.quantization_type == QuantizationType.PER_CHANNEL_ASYMMETRIC
                or self.quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC
            )
        )

    def get_quantization_params(self):
        if not self.is_quantizable():
            return (None, None)
        if not self.quantize_params.zp and not self.quantize_params.scale:
            # Else compute them
            if self.per_channel_quantization:
                num_dims = len(self.np_array.shape)
                num_channels = self.np_array.shape[self.quantize_dim]
                for i in range(num_channels):
                    c = tuple(
                        [
                            i if j == self.quantize_dim else slice(None)
                            for j in range(num_dims)
                        ]
                    )
                    zp, scale = get_quantization_params(
                        self.np_array[c],
                        symmetric=self.symmetric,
                        narrow_range=self.narrow_range,
                        num_quant_bits=self.num_quant_bits,
                    )
                    self.quantize_params.zp.append(zp)
                    self.quantize_params.scale.append(scale)
            else:
                zp, scale = get_quantization_params(
                    self.np_array,
                    symmetric=self.symmetric,
                    narrow_range=self.narrow_range,
                    num_quant_bits=self.num_quant_bits,
                )
                self.quantize_params.zp.append(zp)
                self.quantize_params.scale.append(scale)
        return (self.quantize_params.zp, self.quantize_params.scale)

    def quantize(self):
        if self.quantized:
            return
        if not self.is_quantizable():
            return None
        zp, scale = self.get_quantization_params()
        if self.per_channel_quantization:
            if self.symmetric:
                if self.num_quant_bits == 8:
                    dtype = np.int8
                else:
                    dtype = np.int32
            else:
                if self.num_quant_bits == 8:
                    dtype = np.uint8
                else:
                    dtype = np.uint32

            num_dims = len(self.np_array.shape)
            num_channels = self.np_array.shape[self.quantize_dim]
            q_array = np.zeros(self.np_array.shape, dtype=dtype)
            for i in range(num_channels):
                c = tuple(
                    [
                        i if j == self.quantize_dim else slice(None)
                        for j in range(num_dims)
                    ]
                )
                slc = self.np_array[c]
                if isinstance(slc, np.float32):
                    tmp = np.ndarray((1), dtype=self.np_array.dtype)
                    tmp[0] = slc
                else:
                    tmp = slc
                q = quantize(
                    tmp,
                    zp[i],
                    scale[i],
                    self.symmetric,
                    self.narrow_range,
                    self.num_quant_bits,
                )
                q_array[c] = q
            self.np_array = q_array
        else:
            q = quantize(
                self.np_array,
                zp[0],
                scale[0],
                self.symmetric,
                self.narrow_range,
                self.num_quant_bits,
            )
            self.np_array = q
        self.quantized = True


class Operator:
    def __init__(self, op_type, name, dtypes=None, param_str=None):
        """
        dtypes should be bound to get_dtype methods on a tensor
        """
        if dtypes is None:
            dtypes = []
        self.op_type = op_type
        self.name = name
        self._dtypes = dtypes
        self.param_str = param_str
        self.array_template = env2.get_template("array_template.cpp")
        self.input_map = {}
        self.output_map = {}
        self.ns = ""
        self.type_signature = ""

    @property
    def dtypes(self):
        return [dt() for dt in self._dtypes]

    def set_namespace(self, namespace_str):
        self.ns = namespace_str
        return self

    def set_inputs(self, input_map):
        self.input_map = input_map
        return self

    def set_outputs(self, output_map):
        self.output_map = output_map
        return self

    def render_declaration(self):
        self.type_signature = env2.get_template("op_type_signature.cpp").render(op=self)
        return env2.get_template("declare_operator.cpp").render(op=self)

    def render_eval(self):
        self.type_signature = env2.get_template("op_type_signature.cpp").render(op=self)
        return env2.get_template("eval_operator.cpp").render(op=self)

    def quantize(self):
        for thing in self.input_map:
            self.input_map[thing].quantize()
        for thing in self.output_map:
            self.output_map[thing].quantize()


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
        self.thresholds = []
        for tensor in target_op.input_map:
            self.tensor_set.add(target_op.input_map[tensor])
        for tensor in target_op.output_map:
            self.tensor_set.add(target_op.output_map[tensor])

    def add_tensor_comparison(self, a, b, threshold=0.0):
        """
        if threshold is zero, expect to be exact equal
        """
        self.compare_tensors.append((a, b))
        self.tensor_set.add(a)
        self.tensor_set.add(b)
        self.thresholds.append(threshold)

    def quantize(self):
        self.target_op.quantize()
        # Duplicate quantization because we can
        for (a, b) in self.compare_tensors:
            a.quantize()
            b.quantize()
        for thing in tensor_set:
            thing.quantize()

    def render(self):
        const_snippets = []
        tensor_decls = []
        for tensor in self.tensor_set:
            const_snippets.append(tensor.render_constant())
            tensor_decls.append(tensor.render_declaration())
        op_decl = self.target_op.render_declaration()
        op_eval = self.target_op.render_eval()

        compare_snippets = []
        for (a, b), threshold in zip(self.compare_tensors, self.thresholds):
            compare_snippets.append(
                env2.get_template("compare_outputs.cpp").render(
                    a=a, b=b, threshold=threshold
                )
            )

        TestTemplate = env2.get_template("test_container.cpp")
        test_rendered = TestTemplate.render(
            test_group=self.test_group,
            test_name=self.test_name,
            out_size=self.out_size,
            tensor_declarations=tensor_decls,
            op_decl=op_decl,
            op_eval=op_eval,
            compare_snippets=compare_snippets,
        )
        return (test_rendered, const_snippets)


del _template2_dir
