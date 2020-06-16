from jinja_env import env2, Operator, Tensor, SingleOpTest
import tensorflow as tf

test_group = "Softmax"

def gen_test(test_number, scale = 1.0):
    test_name = "random_gen_scale_%d__%d" % ( int(scale), test_number)
    in1 = tf.constant(tf.random.uniform([1,10])*scale).numpy()
    out_1 = tf.nn.softmax(in1).numpy()

    in_ref_name = "s_ref_in_%d" % test_number
    out_ref_name = "s_ref_out_%d" % test_number
    in_t = Tensor("in", in1, ref_name=in_ref_name)        
    out_ref = Tensor("out_ref", out_1, ref_name=out_ref_name) # Store the reference out values
    out_t = Tensor("out", out_1)                 
    #conv_param_str = "{%s}, %s" % (str(strides).lstrip('[').rstrip(']'), padding)
    #convOp = Operator("Conv2dOperator", "op_0", dtypes=["float"], param_str=conv_param_str)
    op = Operator("SoftmaxOperator", "softmaxOp", dtypes=["float"])
    op.set_inputs({"in": in_t}).set_outputs({"out": out_t})
    
    test = SingleOpTest(test_group, test_name, op)
    test.add_tensor_comparison(out_t, out_ref)
    test_rendered, const_snippets = test.render()
    print(test_rendered)
