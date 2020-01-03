#ifndef UTENSOR_OPERATOR_BASE_H
#define UTENSOR_OPERATOR_BASE_H
#include "TensorMap.hpp"
namespace uTensor{
  // Operators do not go on the heap
  class OperatorBase {
    protected:
      TensorMapInterface* inputs;
      TensorMapInterface* outputs;
    public:
      uTensor::string op_name;
    public:
      OperatorBase(TensorMapInterface* inputs, TensorMapInterface* outputs);
      // The preferred interface
      OperatorBase(TensorMapInterface* inputs);
      OperatorBase();
      void set_inputs(TensorMapInterface* inputs);
      void set_outputs(TensorMapInterface* outputs);
      void set_name(uTensor::string _name);
      virtual ~OperatorBase();

    protected:
      friend class Context;
      virtual void compute() = 0;
  };
}
#endif
