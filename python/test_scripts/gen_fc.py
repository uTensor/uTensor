from jinja_env import env2, Operator, Tensor, SingleOpTest
import tensorflow as tf

test_group = "ReferenceFC"

def gen_test(test_number):
    test_name = "random_gen_fc__%d" % ( test_number)
    in1 = tf.constant(tf.random.uniform([1,2,2,64])).numpy()
    w = tf.constant(tf.random.uniform([512,256])).numpy()
    b = tf.constant(tf.random.uniform([512,1])).numpy()
    # Combine ops to behave like final kernel
    out_1 = tf.math.add(tf.linalg.matmul(w, tf.reshape(in1, (-1,1))), b).numpy()

    in_ref_name = "s_ref_in_%d" % test_number
    w_ref_name = "s_ref_w_%d" % test_number
    b_ref_name = "s_ref_b_%d" % test_number
    out_ref_name = "s_ref_out_%d" % test_number
    in_t = Tensor("in", in1, ref_name=in_ref_name)        
    w_t = Tensor("w", w, ref_name=w_ref_name)        
    b_t = Tensor("b ", b, ref_name=b_ref_name)        
    out_ref = Tensor("out_ref", out_1, ref_name=out_ref_name) # Store the reference out values
    out_t = Tensor("out", out_1)                 
    #conv_param_str = "{%s}, %s" % (str(strides).lstrip('[').rstrip(']'), padding)
    #convOp = Operator("Conv2dOperator", "op_0", dtypes=["float"], param_str=conv_param_str)
    param_str = "Fuseable::NoActivation<float>"
    op = Operator("FullyConnectedOperator", "fcOp", dtypes=["float"], param_str=param_str)
    op.set_inputs({"input": in_t, "filter": w_t, "bias": b_t}).set_outputs({"output": out_t})
    
    test = SingleOpTest(test_group, test_name, op)
    test.add_tensor_comparison(out_t, out_ref)
    test_rendered, const_snippets = test.render()
    return test_rendered, const_snippets
