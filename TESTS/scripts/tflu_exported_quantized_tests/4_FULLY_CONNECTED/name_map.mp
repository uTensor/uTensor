# Key : before -> after
inputs: StatefulPartitionedCall/my_model/dense/Relu -> A
inputs: StatefulPartitionedCall/my_model/dense_1/MatMul/ReadVariableOp/transpose -> filter
inputs: StatefulPartitionedCall/my_model/dense_1/MatMul_bias -> bias
outputs: StatefulPartitionedCall/my_model/dense_1/MatMul_bias -> outpuf_ref
