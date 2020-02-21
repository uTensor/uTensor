#ifndef UTENSOR_OPERATOR_BASE_H
#define UTENSOR_OPERATOR_BASE_H
#include "TensorMap.hpp"
#include "context.hpp"
namespace uTensor {

//EVENTS AND ERRORS
DECLARE_ERROR(OperatorIOSizeMismatchError);


// Operators do not go on the heap
class OperatorBase {
 protected:
  TensorMapInterface* _p_inputs;
  TensorMapInterface* _p_outputs;

 public:
  uTensor::string op_name;

 public:
  // OperatorBase(TensorMapInterface* inputs, TensorMapInterface* outputs);
  // The preferred interface
  OperatorBase();
  OperatorBase(uTensor::string op_name);
  void set_name(uTensor::string _name);
  virtual ~OperatorBase();

  void eval();

 protected:
  OperatorBase(TensorMapInterface* inputs);
  void set_inputs(TensorMapInterface* inputs);
  void set_outputs(TensorMapInterface* outputs);

  friend class Context;
  virtual void compute() = 0;
};

template <size_t num_inputs, size_t num_outputs>
class OperatorInterface : public OperatorBase {
 protected:
  FixedTensorMap<num_inputs> inputs;
  FixedTensorMap<num_outputs> outputs;

 public:
  OperatorInterface() : OperatorBase() {}
  virtual ~OperatorInterface() {}

  // This will throw compile time errors if users provide the wrong number of
  // inputs
  OperatorInterface& set_inputs(FixedTensorMap<num_inputs>&& in) {
    inputs = in;
    OperatorBase::set_inputs(&inputs);
    return *this;
  }
  OperatorInterface& set_outputs(FixedTensorMap<num_outputs>&& out) {
    outputs = out;
    OperatorBase::set_outputs(&outputs);
    return *this;
  }

 protected:
  virtual void compute() = 0;
};

// Just a tag for Tensor access power
class FastOperator {
 public:
  size_t get_readable_block(const Tensor& t, void* buffer, uint16_t num_elems,
                            int linear_index);
  size_t get_writeable_block(Tensor& t, void* buffer, uint16_t num_elems,
                             int linear_index);
};
}  // namespace uTensor
#endif
