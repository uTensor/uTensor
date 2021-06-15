# Model Interface (TanhModel)

Simple example of `ModelInterface`:

```cpp
// TanhModel is a model with 1 input tensor and 1 output tensor
class TanhModel : public ModelInterface<1, 1> 
{
 public:
  // input tensor names
  enum input_names : uint8_t { input_0 };
  // output tensor names
  enum output_names : uint8_t { output_0 };
  ...
}
```

## Set Inputs/Outputs and Eval

```cpp
TanhModel tanh_model;
Tensor in_tensor = ...;
Tensor out_tensor = ...;
...
tanh_model
  .set_inputs({
      {TanhModel::input_0, in_tensor}
  })
  .set_outputs({
      {TanhModel::output_0, out_tensor}
  })
  .eval();
...
```
