# Key : before -> after
inputs: input_1_int8 -> A
inputs: StatefulPartitionedCall/my_model/conv2d/Conv2D/ReadVariableOp -> filter
inputs: StatefulPartitionedCall/my_model/conv2d/Conv2D_bias -> bias
outputs: StatefulPartitionedCall/my_model/conv2d/Relu -> output_ref
