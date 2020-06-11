# Writing Custom Operator

All operators in `uTensor` share the same interface, `OperatorInterface`, which is defined and implemented
with debugablity and efficiency in mind.

As metioned in the high-level `uTensor` [documentation](../../src/uTensor/README.md#OperatorInterface), `OperatorInterface` is defined with fixed-size `TensorMap`s for inputs and outputs, setter api for inputs and outputs and pure virtual method `compute` which get invoked whenever `eval` invoked. As a result, to implement
and eval a custom operator, users only need to do as following:

1. implement `compute` method
2. give inputs/outputs with explicit names, best done with two sets of enums respectively
3. invoke `eval`

Also, the number of inputs/outputs is checked at compile time. You only need to provide the number of inputs/ouptuts in the declaration of the operator specified by the template parameters:

```cpp
// an operator with 2 input tensors and 1 output tensors
class MyOperator : public OperatorInterface<2, 1> {
    ....
};
```

For this tutorial, we will implement a very simple operator, `MyOperator`, which just adds up two input tensors and writes the results to an output tensor.

## Implementation Convention

To improve code interoperability, `uTensor` operators are generally implemented as a user-facing implementation of `OperatorInterface` paired with lower-level kernel functions which will be invoked via the `compute` method. Such kernel functions are plain C++, and possibly C, functions. This way, operators with the same high level functional behavior can share user interfaces, but can target different, and potentially optimized, kernels.

For the detail code snippet, please refer to [custom_operator.cpp](custom_operator.cpp).