# Owners

Neil Tan, (more)

# Description

This model generates the unity-testing data for Ops used in deep_mlp.py that is not already covered in qntMatMulDequnt

See: https://github.com/neil-tan/tf-node-viewer

Used for:

- Debug data generation
- Ops covered: Reshape, Add, QuantizedRelu, ArgMax
- Test data for Max and Min will be extracted from deep_mlp.py
# Useful Commands

    ipython3
    run miscOps.py
    import view_node as nv
    nv.init(tf.get_default_graph)
    nv.ls()
    with tf.Session() as sess:
         nv.snap("ref_reshape")
         nv.snap("ref_min")
         nv.snap("ref_max")
         nv.snap("ref_qRelu")
         nv.snap("ref_add")

# TODO
TBD

# Reference

- Reshape:
    - reshape_op.h:90

- QuantizedRelu:
    - quantized_activation_ops.cc:47
    - gen_nn_ops.py:2169

- ArgMax
    - math_ops.py: 175

- Add
    - gen_math_ops.py: 53
