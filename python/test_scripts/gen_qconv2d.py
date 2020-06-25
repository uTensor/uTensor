from jinja_env import env2, Operator, Tensor, SingleOpTest, QuantizationType
import tensorflow as tf
import numpy as np
import copy

test_group = "Conv2D"
num_tests = 2;
output_file = "test_sq_conv2d.cpp"
const_file = "constants_sq_conv2d.hpp"

def gen_test(test_number):
    test_name = "random_gen_conv2d__%d" % ( test_number)
    in0 = np.random.uniform(low=-5, high=5, size=[1,14,14,32]).astype(np.float32)
    w0 = np.random.uniform(low=-1, high=1, size=[3,3,32, 64]).astype(np.float32)
    b = np.random.uniform(low=-2.2, high=2.2, size=(1,64)).astype(np.float32).flatten()
    m = tf.nn.conv2d(in0, w0, strides=[1,1,1,1], padding="VALID")
    print(m.shape)
    out_1 = tf.math.add(m, b).numpy()
    
    # Update Weights to match TFLu [64,3,3,32]
    w = np.zeros((w0.shape[3], w0.shape[0], w0.shape[1], w0.shape[2]), dtype=w0.dtype)
    for i0 in range(w.shape[0]):
      for i1 in range(w.shape[1]):
        for i2 in range(w.shape[2]):
          for i3 in range(w.shape[3]):
            w[i0, i1, i2, i3] = w0[i1, i2, i3, i0]

    # Generate float Tests!
    in_ref_name = "s_ref_in_%d_%s"   % (test_number, "f")
    w_ref_name = "s_ref_w_%d_%s"     % (test_number, "f")
    b_ref_name = "s_ref_b_%d_%s"     % (test_number, "f")
    out_ref_name = "s_ref_out_%d_%s" % (test_number, "f")
    in_t = Tensor("in", in0, ref_name=in_ref_name, quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC)
    w_t = Tensor("w", w, ref_name=w_ref_name, quantization_type=QuantizationType.PER_CHANNEL_SYMMETRIC, quantize_dim= 0, narrow_range=True)
    b_t = Tensor("b", b, ref_name=b_ref_name, quantization_type=QuantizationType.PER_CHANNEL_SYMMETRIC, quantize_dim=0, num_quant_bits=32)
    out_ref = Tensor("out_ref", out_1, ref_name=out_ref_name, quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC) # Store the reference out values
    out_t = Tensor("out", out_1, quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC)                 
    #conv_param_str = "{%s}, %s" % (str(strides).lstrip('[').rstrip(']'), padding)
    #convOp = Operator("Conv2dOperator", "op_0", dtypes=["float"], param_str=conv_param_str)
    #param_str = "Fuseable::NoActivation<float>"
    param_str = "{1,1,1,1}, SAME"
    op = Operator("Conv2dOperator", "convOp", dtypes=[in_t.get_dtype], param_str=param_str)
    op.set_inputs({"input": in_t, "filter": w_t, "bias": b_t}).set_outputs({"output": out_t})
    
    test = SingleOpTest(test_group, test_name + "_f", op)
    test.add_tensor_comparison(out_t, out_ref)
    test_rendered_f, const_snippets_f = test.render()
    
    # Quantize!
    in_ref_name = "s_ref_in_%d_%s"   % (test_number, "q")
    in_ref_name = "s_ref_in_%d_%s"   % (test_number, "q")
    w_ref_name = "s_ref_w_%d_%s"     % (test_number, "q")
    b_ref_name = "s_ref_b_%d_%s"     % (test_number, "q")
    out_ref_name = "s_ref_out_%d_%s" % (test_number, "q")
    out_name = "s_out_%d_%s" % (test_number, "q")
    in_t.ref_name = in_ref_name
    w_t.ref_name = w_ref_name
    b_t.ref_name = b_ref_name
    out_ref.ref_name = out_ref_name
    out_t.quantize_params.ref_name = out_name # Gotta give it somewhere to store params
    in_t.quantize()
    w_t.quantize()
    # Bias quantization params depend on input and wieghts
    b_t.quantize_params.scale =  [in_t.quantize_params.scale[0]*j for j in w_t.quantize_params.scale]
    b_t.quantize_params.zp = copy.deepcopy(w_t.quantize_params.zp)
    b_t.quantize()
    out_t.quantize()
    out_ref.quantize()
    
    test = SingleOpTest(test_group, test_name + "_q", op)
    test.add_tensor_comparison(out_t, out_ref)
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
        gt_r = env2.get_template("gtest_container.cpp").render(constants_header=const_file, using_directives=["using namespace uTensor::ReferenceOperators"], tests=tests)
        fp.write(gt_r)
