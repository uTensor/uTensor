#ifndef UTENSOR_OPERATOR_BASE_H
#define UTENSOR_OPERATOR_BASE_H
#include "TensorMap.hpp"
namespace uTensor{
// Operators do not go on the heap
class OperatorBase {
  protected:
    TensorMapInterface* _p_inputs;
    TensorMapInterface* _p_outputs;
  public:
    uTensor::string op_name;
  public:
    //OperatorBase(TensorMapInterface* inputs, TensorMapInterface* outputs);
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
    virtual ~OperatorInterface();
    OperatorInterface() : OperatorBase() {}

    // This will throw compile time errors if users provide the wrong number of inputs
    void set_inputs(FixedTensorMap<num_inputs>&& in) {
      inputs = in;
      OperatorBase::set_inputs(&inputs);
    }
    void set_outputs(FixedTensorMap<num_outputs>&& out) {
      outputs = out;
      OperatorBase::set_outputs(&outputs);
    }

  protected:
    virtual void compute() = 0;
};
}
#endif
