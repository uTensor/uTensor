import tensorflow as tf
import argparse
from jinja_env import env
import numpy as np


def main(cpp_fname, const_fname, is_argmin):
    # template render variables
    utensor_headers = set(["ArgMinMax.hpp", "RamTensor.hpp", "RomTensor.hpp"])
    test_headers = set([const_fname])
    constants_map = {}
    test_suit_name = is_argmin and "ArgMin" or "ArgMax"
    test_name = is_argmin and "random_argmin_test" or "random_argmax_test"
    output_size = 10
    declare_tensor_strs = []
    op_cls = is_argmin and "ArgMinOperator" or "ArgMaxOperator"
    op_type_signature = "float"
    op_name = "op"
    op_construct_params = []
    inputs_str = ""
    outputs_str = ""
    output_names = []
    ref_output_names = []
    # generate testing data
    tensor_input = tf.random.uniform((10, 5), maxval=5, dtype=tf.float32)
    np_input = tensor_input.numpy()
    tf_op = [tf.argmax, tf.argmin][is_argmin]
    tensor_output = tf_op(tensor_input, axis=1)
    np_output = tensor_output.numpy()

    constants_map["random_input_arr"] = (np_input.flatten().tolist(), "float")
    constants_map["const_axis"] = ([1], "uint32_t")
    constants_map["ref_output_arr"] = (np_output.flatten().tolist(), "uint32_t")
    declare_tensor_strs.extend(
        [
            env.get_template("declare_rom_tensor.cpp").render(
                tensor_name="input_tensor",
                shape=np_input.shape,
                tensor_type_str="float",
                const_var_name="random_input_arr",
            ),
            env.get_template("declare_rom_tensor.cpp").render(
                tensor_name="axis_tensor",
                shape=[1],
                tensor_type_str="uint32_t",
                const_var_name="const_axis",
            ),
            env.get_template("declare_ram_tensor.cpp").render(
                tensor_name="output_tensor",
                shape=np_output.shape,
                tensor_type_str="uint32_t",
            ),
        ]
    )
    inputs_str += f"{{ {op_cls}<{op_type_signature}>::input, input_tensor }}, "
    inputs_str += f"{{ {op_cls}<{op_type_signature}>::axis, axis_tensor }}"
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
                output_size=np_output.flatten().shape[0],
                declare_tensor_strs=declare_tensor_strs,
                op_cls=op_cls,
                op_type_signature=op_type_signature,
                op_name=op_name,
                inputs_str=inputs_str,
                outputs_str=outputs_str,
                output_names=output_names,
                ref_output_names=ref_output_names,
                output_type_str="uint32_t",
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
        default="test_arg_min_max.cpp",
    )
    parser.add_argument(
        "--const-fname",
        help="the header file containing constants for test (default: %(default)s)",
        metavar="CONST.hpp",
        default="const_arg_min_max.hpp",
    )
    parser.add_argument(
        "--argmin", dest="is_argmin", action="store_true", help="generate ArgMin test"
    )
    args = parser.parse_args()
    main(**vars(args))
