import argparse
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np

from jinja_env import env

TFLM_Tensor = namedtuple(
    "TFLM_Tensor", ["tensor", "quantization"]
)  # it's required for loading the pickle files


def quantize(arr, zero_point, scale):
    # the spec: https://www.tensorflow.org/lite/performance/quantization_spec
    return (np.round(arr / scale) + zero_point).astype("int8")


def main(
    cpp_fname, const_fname, test_data_dir="./tflu_exported_quantized_tests/0_QUANTIZE"
):
    # load testing data
    test_data_dir = Path(test_data_dir)
    with (test_data_dir / "inputs.pkl").open("rb") as input_fid, (
        test_data_dir / "outputs.pkl"
    ).open("rb") as output_fid:
        outputs_quant_param = pickle.load(output_fid)["input_1_int8"].quantization
        input_tensor = pickle.load(input_fid)["input_1"].tensor
        quant_tensor = quantize(
            input_tensor,
            zero_point=outputs_quant_param[1],
            scale=outputs_quant_param[0],
        )

    # template render variables
    utensor_headers = set(["QuantizeOps.hpp", "RamTensor.hpp", "RomTensor.hpp"])
    test_headers = set([const_fname])
    constants_map = {}
    test_suit_name = "Quantized"
    test_name = "reference_0_quantize"
    output_size = quant_tensor.size
    declare_tensor_strs = []
    op_cls = "::TFLM::QuantizeOperator"
    op_type_signature = "int8_t, float"
    op_name = "op"
    op_construct_params = []
    inputs_str = ""
    outputs_str = ""
    output_names = []
    ref_output_names = []

    constants_map["input_arr"] = (input_tensor.flatten().tolist(), "float")
    constants_map["ref_output_arr"] = (quant_tensor.flatten().tolist(), "int8_t")
    declare_tensor_strs.extend(
        [
            env.get_template("declare_rom_tensor.cpp").render(
                tensor_name="input_tensor",
                shape=input_tensor.shape,
                tensor_type_str="float",
                const_var_name="input_arr",
            ),
            env.get_template("declare_ram_tensor.cpp").render(
                tensor_name="output_tensor",
                shape=quant_tensor.shape,
                tensor_type_str="int8_t",
                quantize_params=outputs_quant_param,
            ),
        ]
    )
    inputs_str += f"{{ {op_cls}<{op_type_signature}>::input, input_tensor }}"
    outputs_str += f"{{ {op_cls}<{op_type_signature}>::output, output_tensor }}"
    output_names.append("output_tensor")
    ref_output_names.append("ref_output_arr")

    # render templates
    test_template = env.get_template("test_container.cpp")
    const_template = env.get_template("test_const.hpp")
    with open(cpp_fname, "w") as cpp_fid, open(const_fname, "w") as header_fid:
        cpp_fid.write(
            test_template.render(
                test_suit_name=test_suit_name,
                test_name=test_name,
                utensor_headers=utensor_headers,
                test_headers=test_headers,
                output_size=quant_tensor.size,
                declare_tensor_strs=declare_tensor_strs,
                op_cls=op_cls,
                op_type_signature=op_type_signature,
                op_construct_params=op_construct_params,
                op_name=op_name,
                inputs_str=inputs_str,
                outputs_str=outputs_str,
                output_names=output_names,
                ref_output_names=ref_output_names,
                output_type_str="int8_t",
            )
        )
        header_fid.write(
            const_template.render(constants_map=constants_map, test_name=test_name)
        )
    print(f"generating output files: {cpp_fname}, {const_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpp-fname",
        help="the output cpp file name (default: %(default)s)",
        metavar="TEST.cpp",
        default="test_quantize.cpp",
    )
    parser.add_argument(
        "--const-fname",
        help="the header file containing constants for test (default: %(default)s)",
        metavar="CONST.hpp",
        default="constants_quantize.hpp",
    )
    parser.add_argument(
        "--test-data-dir",
        help="the directory of testing data (default: %(default)s)",
        metavar="DIR",
        default="tflu_exported_quantized_tests/0_QUANTIZE",
    )
    args = parser.parse_args()
    main(**vars(args))
