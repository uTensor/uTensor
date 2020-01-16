#ifndef UTENSOR_MODEL_BASE_H
#define UTENSOR_MODEL_BASE_H
#include "TensorMap.hpp"
namespace uTensor{
// Models do not go on the heap
class ModelBase {
  protected:
    TensorMapInterface* _p_inputs;
    TensorMapInterface* _p_outputs;
  public:
    uTensor::string _name;
  public:
    //ModelBase(TensorMapInterface* inputs, TensorMapInterface* outputs);
    // The preferred interface
    ModelBase();
    ModelBase(uTensor::string _name);
    void set_name(uTensor::string _name);
    virtual ~ModelBase();

    void eval();
  protected:
    ModelBase(TensorMapInterface* inputs);
    void set_inputs(TensorMapInterface* inputs);
    void set_outputs(TensorMapInterface* outputs);

    friend class Context;
    virtual void compute() = 0;
};

template <size_t num_inputs, size_t num_outputs>
class ModelInterface : public ModelBase {
  protected:
    FixedTensorMap<num_inputs> inputs;
    FixedTensorMap<num_outputs> outputs;
  public:
    ModelInterface() : ModelBase() {}
    virtual ~ModelInterface() {}

    // This will throw compile time errors if users provide the wrong number of inputs
    ModelInterface& set_inputs(FixedTensorMap<num_inputs>&& in) {
      inputs = in;
      ModelBase::set_inputs(&inputs);
      return *this;
    }
    ModelInterface& set_outputs(FixedTensorMap<num_outputs>&& out) {
      outputs = out;
      ModelBase::set_outputs(&outputs);
      return *this;
    }

  protected:
    virtual void compute() = 0;
};
}
#endif
