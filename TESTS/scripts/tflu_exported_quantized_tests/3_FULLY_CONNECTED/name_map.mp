# Key : before -> after
inputs: StatefulPartitionedCall/my_model/max_pooling2d/MaxPool -> A
inputs: StatefulPartitionedCall/my_model/dense/MatMul/ReadVariableOp/transpose -> filter
inputs: StatefulPartitionedCall/my_model/dense/MatMul_bias -> bias
outputs: StatefulPartitionedCall/my_model/dense/MatMul_bias -> output_ref
