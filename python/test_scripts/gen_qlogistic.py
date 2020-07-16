from jinja_env import env2, Operator, Tensor, SingleOpTest, QuantizationType
import tensorflow as tf
import numpy as np
import copy

test_group = "Logistic"
num_tests = 4
test_base = "logistic"
output_file = "test_sq_%s.cpp" % test_base
const_file = "constants_sq_%s.hpp" % test_base
operator_name = "LogisticOperator"
operator_var = "logisticOp"

def gen_test(test_number):
    test_name = "random_gen_%s__%d" % ( test_base, test_number)
    in0 = np.random.uniform(low=-20, high=50, size=(1,64)).astype(np.float32).flatten()
    out_1 = tf.nn.sigmoid(in0).numpy()

    # Generate float Tests!
    in_ref_name = "s_ref_in_%d_%s"   % (test_number, "f")
    out_ref_name = "s_ref_out_%d_%s" % (test_number, "f")
    in_t = Tensor("in", in0, ref_name=in_ref_name, quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC)
    out_ref = Tensor("out_ref", out_1, ref_name=out_ref_name, quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC) # Store the reference out values
    out_t = Tensor("out", out_1, quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC)                 
    
    op = Operator(operator_name, operator_var, dtypes=[in_t.get_dtype])
    op.set_inputs({"in": in_t}).set_outputs({"out": out_t})
    op.set_namespace("uTensor::ReferenceOperators::")
    
    test = SingleOpTest(test_group, test_name + "_f", op)
    test.add_tensor_comparison(out_t, out_ref, threshold=0.0001)
    test_rendered_f, const_snippets_f = test.render()
    
    # Quantize!
    in_ref_name  = "s_ref_in_%d_%s"   % (test_number, "q")
    out_ref_name = "s_ref_out_%d_%s" % (test_number, "q")
    out_name     = "s_out_%d_%s" % (test_number, "q")
    in_t.ref_name = in_ref_name
    out_ref.ref_name = out_ref_name
    out_t.quantize_params.ref_name = out_name # Gotta give it somewhere to store params
    in_t.quantize()
    # output quantization params are fixed
    out_t.quantize_params.scale =  [1.0/256.0]
    out_t.quantize_params.zp = [-128]
    out_t.quantized = True
    out_ref.quantize_params.scale = copy.deepcopy(out_t.quantize_params.scale)
    out_ref.quantize_params.zp = copy.deepcopy(out_t.quantize_params.zp)
    out_ref.quantized = True
    #op.set_namespace("uTensor::ReferenceOperators::")
    
    test = SingleOpTest(test_group, test_name + "_q", op)
    test.add_tensor_comparison(out_t, out_ref, threshold=2)
    test_rendered_q, const_snippets_q = test.render()
    
    return [(test_rendered_f, const_snippets_f),(test_rendered_q, const_snippets_q)]


if __name__ == '__main__':
    tests = []
    const_snippets =[]
    for i in range(num_tests):
        rendered = gen_test(i)
        for tr, cs in rendered:
            tests.append(tr)
            const_snippets.extend(cs)
    with open(const_file, "w") as fp:
        c_r = env2.get_template("const_container.hpp").render(constants=const_snippets, constants_header=const_file)
        fp.write(c_r)
    with open(output_file, "w") as fp:
        gt_r = env2.get_template("gtest_container.cpp").render(constants_header=const_file, using_directives=[], tests=tests)
        fp.write(gt_r)
