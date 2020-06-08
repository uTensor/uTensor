# Writing Custom Operator

All operators in `uTensor` share the same interface, `OperatorInterface`, which is define and implemented
with debugablity and efficiency in mind.

As metioned in the high-level `uTensor` [documentation](../../src/uTensor/README.md#OperatorInterface), `OperatorInterface` is defined with fixed-size `TensorMap`s for inputs and outputs, setter api for inputs and outputs and pure virtual method `compute` which get invoked whenever `eval` invoked. As a result, to implement
and eval a custom operator, users only need to do as following:

1. implement `compute` method
2. setting inputs/outputs with explicit names (of type `uTensor::string` which is implicitly converted to an integer)
3. invoke `eval`

Also, the number of inputs/outputs is checked at compile time. You only need to provide the number of inputs/ouptuts in the declaration of the operator specified by the template parameters:

```cpp
// an operator with 2 input tensors and 1 output tensors
class MyOperator : public OperatorInterface<2, 1> {
    ....
};
```

To keep things simple, we will implement a very simple operator, `MyOperator`, which simply add up two input tensors and write the results to a output tensor.

## Implementation Convention

For better code reusability, `uTensor` operators are normally implemented with kernel functions which will be invoked in the `compute` method. Such kernel functions are plain c++ functions. This way, multiple operators can share kernels more easily. In this tutorial, we will follow such convention. However, it's not required for implementing operators in `uTensor`.

For the detail code snippet, please refer to [custom_operator.cpp](custom_operator.cpp).