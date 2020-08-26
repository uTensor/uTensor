from copy import deepcopy

import numpy as np
import tensorflow as tf

from jinja_env import Operator, QuantizationType, SingleOpTest, Tensor
from jinja_env.quantization_util import quantize
from model_base import ModelBase


class ReduceMeanModel(tf.keras.Model):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    @tf.function
    def call(self, x):
        return tf.math.reduce_mean(x, axis=self.axis)


class QuantReduceMeanModel(ModelBase):

    TEST_GROUP = "QuantReduceMean"

    def __init__(self, shape, axis=None):
        model = ReduceMeanModel(axis)
        x = np.random.randn(1, *shape).astype("float32")
        model._set_inputs(x)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.representative_dataset = self.representative_dataset(shape)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self._model_content = converter.convert()
        self._shape = shape
        self.axis = axis

    def generate_test_case(self, test_case_name):
        interpreter = tf.lite.Interpreter(model_content=self._model_content)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        in_values = np.random.randn(1, *self._shape).astype("float32")
        interpreter.set_tensor(input_info["index"], in_values)
        interpreter.invoke()
        out_values = np.atleast_1d(interpreter.tensor(output_info["index"])())
        quant_input_info, quant_output_info = self.get_quant_infos(interpreter)
        quant_in_values = quantize(
            in_values,
            zp=quant_input_info["quantization"][1],
            scale=quant_input_info["quantization"][0],
            symmetric=True,
        )
        quant_out_values = quantize(
            out_values,
            zp=quant_output_info["quantization"][1],
            scale=quant_output_info["quantization"][0],
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
        in_tensor.quantize_params.zp = [quant_input_info["quantization"][1]]
        in_tensor.quantize_params.scale = [quant_input_info["quantization"][0]]
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
        out_tensor.quantize_params.zp = [quant_output_info["quantization"][1]]
        out_tensor.quantize_params.scale = [quant_output_info["quantization"][0]]
        out_tensor.quantize_params.ref_name = out_ref_name
        out_tensor.quantized = True
        rank = len(self._shape) + 1
        if self.axis is None:
            param_str = "{" + ", ".join([str(i) for i in range(rank)]) + "}"
        else:
            param_str = "{" + ", ".join(map(str, self.axis)) + "}"
        op = Operator(
            "ReduceMeanOperator",
            "mean_op",
            dtypes=[lambda: "int8_t"],
            param_str=param_str,
        )
        op.set_inputs({"input": in_tensor}).set_outputs({"output": out_tensor})
        op.set_namespace("uTensor::ReferenceOperators::")
        test = SingleOpTest(self.TEST_GROUP, test_case_name, op)
        test.add_tensor_comparison(out_tensor, ref_out_tensor, 2)
        return test.render()

    @staticmethod
    def get_quant_infos(interpreter):
        node_idx = None
        for i in range(interpreter._interpreter.NumNodes()):
            if interpreter._interpreter.NodeName(i) == "MEAN":
                node_idx = i
                break
        else:
            raise ValueError("MEAN node not found")
        input_idx = interpreter._interpreter.NodeInputs(node_idx)[0]
        output_idx = interpreter._interpreter.NodeOutputs(node_idx)[0]
        quant_input_info = None
        quant_output_info = None
        for info in interpreter.get_tensor_details():
            if info["index"] == input_idx:
                quant_input_info = deepcopy(info)
            elif info["index"] == output_idx:
                quant_output_info = deepcopy(info)
        return quant_input_info, quant_output_info

    @staticmethod
    def representative_dataset(shape):
        def dataset():
            for _ in range(5000):
                yield [np.random.randn(1, *shape).astype("float32")]

        return dataset


def main():
    shape = (3, 5, 7)
    model = QuantReduceMeanModel(shape)
    model.render_files("reduce_mean_no_axis")
    model = QuantReduceMeanModel(shape, [0])
    model.render_files("reduce_mean_axis_first")
    model = QuantReduceMeanModel(shape, [3])
    model.render_files("reduce_mean_axis_last")
    model = QuantReduceMeanModel(shape, [2])
    model.render_files("reduce_mean_axis_2")
    model = QuantReduceMeanModel(shape, [0, 2])
    model.render_files("reduce_mean_axis_0_2")
    model = QuantReduceMeanModel(shape, [2, 3])
    model.render_files("reduce_mean_axis_2_3")


if __name__ == "__main__":
    main()
