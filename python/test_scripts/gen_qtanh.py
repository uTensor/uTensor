import argparse
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf

from jinja_env import Operator, QuantizationType, SingleOpTest, Tensor, env2
from jinja_env.quantization_util import quantize


class ModelBase:
    def generate_test_case(self, test_case_name):
        raise NotImplementedError("not implemented")

    def generate_test_cases(self, test_name, num_tests):
        for i in range(num_tests):
            yield self.generate_test_case(f"{test_name}_{i}")

    def render_files(self, test_name, num_tests=5, const_fname=None, src_fname=None):
        if const_fname is None:
            const_fname = f"constants_{test_name}.hpp"
        if src_fname is None:
            src_fname = f"test_{test_name}.cpp"
        cases = self.generate_test_cases(test_name, num_tests)
        const_snippets = []
        test_snippets = []
        for ts, cs in cases:
            const_snippets.extend(cs)
            test_snippets.append(ts)
        with open(const_fname, "w") as fid:
            print(f"generating {const_fname}")
            fid.write(
                env2.get_template("const_container.hpp").render(
                    constants=const_snippets, constants_header=const_fname
                )
            )
        with open(src_fname, "w") as fid:
            print(f"generating {src_fname}")
            fid.write(
                env2.get_template("gtest_container.cpp").render(
                    constants_header=const_fname,
                    using_directives=[],
                    tests=test_snippets,
                )
            )


class QuantTanhModel(ModelBase):

    TEST_GROUP = "QuantTanhTest"

    def __init__(self, tflite_file):
        with open(tflite_file, "rb") as fid:
            self._model_content = fid.read()

    def generate_test_case(self, test_case_name):
        self.interpreter = tf.lite.Interpreter(model_content=self._model_content)
        self.interpreter.allocate_tensors()
        in_values = np.random.rand(*self.in_dim).astype("float32")
        self.interpreter.set_tensor(self.input_idx, in_values)
        self.interpreter.invoke()
        out_values = self.interpreter.tensor(self.output_idx)()
        quant_in_values = quantize(
            in_values,
            zp=self.quant_input_info["quantization"][1],
            scale=self.quant_input_info["quantization"][0],
            symmetric=True,
        )
        quant_out_values = quantize(
            out_values,
            zp=self.quant_output_info["quantization"][1],
            scale=self.quant_output_info["quantization"][0],
            symmetric=True,
        )
        in_ref_name = f"s_ref_input_{test_case_name}"
        out_ref_name = f"s_ref_output_{test_case_name}"
        in_tensor = Tensor(
            "input",
            quant_in_values,
            ref_name=in_ref_name,
            quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC,
        )
        in_tensor.quantize_params.zp = [self.quant_input_info["quantization"][1]]
        in_tensor.quantize_params.scale = [self.quant_input_info["quantization"][0]]
        in_tensor.quantized = True
        ref_out_tensor = Tensor(
            "ref_output",
            quant_out_values,
            ref_name=out_ref_name,
            quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC,
        )
        out_tensor = Tensor(
            "output",
            quant_out_values,
            quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC,
        )
        out_tensor.quantize_params.zp = [self.quant_output_info["quantization"][1]]
        out_tensor.quantize_params.scale = [self.quant_output_info["quantization"][0]]
        out_tensor.quantize_params.ref_name = out_ref_name
        out_tensor.quantized = True
        op = Operator(
            "TanhOperator", "tanh_op", dtypes=[lambda: "int8_t", lambda: "int8_t"],
        )
        op.set_inputs({"act_in": in_tensor}).set_outputs({"act_out": out_tensor})
        op.set_namespace("uTensor::ReferenceOperators::")
        test = SingleOpTest(self.TEST_GROUP, test_case_name, op)
        test.add_tensor_comparison(out_tensor, ref_out_tensor, 2)
        return test.render()

    @property
    def interpreter(self):
        return self._interpreter

    @interpreter.setter
    def interpreter(self, it):
        self._interpreter = it
        node_idx = 0
        for i in range(it._interpreter.NumNodes()):
            if it._interpreter.NodeName(i) == "TANH":
                node_idx = i
                break
        else:
            raise ValueError("TANH node not found")
        input_idx = it._interpreter.NodeInputs(node_idx)[0]
        output_idx = it._interpreter.NodeOutputs(node_idx)[0]
        for info in it.get_tensor_details():
            if info["index"] == input_idx:
                self._quant_input_info = deepcopy(info)
            elif info["index"] == output_idx:
                self._quant_output_info = deepcopy(info)

    @property
    def in_dim(self):
        return self._interpreter.get_input_details()[0]["shape"].tolist()

    @property
    def input_idx(self):
        input_info = self._interpreter.get_input_details()[0]
        return input_info["index"]

    @property
    def output_idx(self):
        out_info = self._interpreter.get_output_details()[0]
        return out_info["index"]

    @property
    def quant_input_info(self):
        return self._quant_input_info

    @property
    def quant_output_info(self):
        return self._quant_output_info


def main(model_path, num_tests=5):
    model = QuantTanhModel(model_path)
    model.render_files("sq_tanh", num_tests=num_tests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num-tests",
        metavar="INT",
        type=int,
        dest="num_tests",
        default=5,
        help="the number of test cases [default: %(default)s]",
    )
    parser.add_argument(
        "model_path",
        metavar="MODEL.tflite",
        help="the model file for generating tests [default: %(default)s]",
    )
    args = vars(parser.parse_args())
    main(**args)
