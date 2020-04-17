import argparse
from functools import reduce

import numpy as np
import tensorflow as tf

from jinja_env import env


def main(cpp_fname, const_fname):
    # template render variables
    utensor_headers = set(["Reshape.hpp", "RamTensor.hpp", "RomTensor.hpp"])
    test_headers = set([const_fname])
    constants_map = {}
    test_suit_name = "Reshape"
    test_name = "reshape_test"
    output_size = 10
    declare_tensor_strs = []
    op_cls = "ReshapeOperator"
    op_type_signature = "float"
    op_name = "op"
    op_construct_params = []
    inputs_str = ""
    outputs_str = ""
    output_names = []
    ref_output_names = []
    other_tests_str = []

    # generate testing data
    tensor_input = tf.random.uniform((3, 5), maxval=5, dtype=tf.float32)
    np_input = tensor_input.numpy()
    new_shape = [5, 3, 1]
    op_construct_params.append("{{ {} }}".format(",".join(map(str, new_shape))))
    tensor_output = tf.reshape(tensor_input, new_shape)
    np_output = tensor_output.numpy()

    constants_map["random_input_arr"] = (np_input.flatten().tolist(), "float")
    constants_map["ref_output_arr"] = (np_output.flatten().tolist(), "float")
    declare_tensor_strs.extend(
        [
            env.get_template("declare_rom_tensor.cpp").render(
                tensor_name="input_tensor",
                shape=np_input.shape,
                tensor_type_str="float",
                const_var_name="random_input_arr",
            ),
            env.get_template("declare_ram_tensor.cpp").render(
                tensor_name="output_tensor", tensor_type_str="float",
            ),
        ]
    )
    inputs_str += f"{{ {op_cls}<{op_type_signature}>::input, input_tensor }}"
    outputs_str += f"{{ {op_cls}<{op_type_signature}>::output, output_tensor }}"
    output_names.append("output_tensor")
    ref_output_names.append("ref_output_arr")
    other_tests_str.append(
        f"TensorShape target_shape({ ','.join(map(str, new_shape)) });\n"
        "  TensorShape output_shape = output_tensor->get_shape();\n"
        "  EXPECT_TRUE(target_shape == output_shape);\n"
    )

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
                output_size=np_output.size,
                declare_tensor_strs=declare_tensor_strs,
                op_cls=op_cls,
                op_type_signature=op_type_signature,
                op_construct_params=op_construct_params,
                op_name=op_name,
                inputs_str=inputs_str,
                outputs_str=outputs_str,
                output_names=output_names,
                ref_output_names=ref_output_names,
                output_type_str="float",
                tol=0.0001,
                other_tests_str=other_tests_str,
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
        default="test_reshape.cpp",
    )
    parser.add_argument(
        "--const-fname",
        help="the header file containing constants for test (default: %(default)s)",
        metavar="CONST.hpp",
        default="constants_reshape.hpp",
    )
    args = parser.parse_args()
    main(**vars(args))
