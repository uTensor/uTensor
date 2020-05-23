#ifndef UTENSOR_ACTIVATIONS_OPS_H
#define UTENSOR_ACTIVATIONS_OPS_H
#include <type_traits>

#include "ActivationFncs_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {
class InPlaceActivationFnc : public OperatorInterface<1, 0> {
 public:
  enum names_in : uint8_t { x };

 protected:
  virtual void compute() = 0;
};

template <typename T>
class InPlaceReLU : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 protected:
  virtual void compute() { inplace_relu_k<T>(inputs[x].tensor()); }
};

template <typename T>
class ReLUOperator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    const Tensor& inT = inputs[in].tensor();
    Tensor& outT = outputs[out].tensor();
    // TODO Check sizes here and throw mismatch
    uint32_t in_size = inT->get_shape().get_linear_size();
    uint32_t out_size = outT->get_shape().get_linear_size();
    if (in_size != out_size)
      Context::get_default_context()->throwError(
          new OperatorIOSizeMismatchError);
    relu_k<T>(outT, inT);
  }
};

template <typename T>
class InPlaceReLU6 : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 protected:
  virtual void compute() { inplace_relu6_k<T>(inputs[x].tensor()); }
};

template <typename T>
class ReLU6Operator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    const Tensor& inT = inputs[in].tensor();
    Tensor& outT = outputs[out].tensor();
    // TODO Check sizes here and throw mismatch
    uint32_t in_size = inT->get_shape().get_linear_size();
    uint32_t out_size = outT->get_shape().get_linear_size();
    if (in_size != out_size)
      Context::get_default_context()->throwError(
          new OperatorIOSizeMismatchError);
    relu6_k<T>(outT, inT);
  }
};

template <typename T>
class InPlaceSoftmax : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct Softmax on non-signed types");

 public:
  InPlaceSoftmax() : beta(1) {}
  InPlaceSoftmax(T beta) : beta(beta) {}
 protected:
  virtual void compute() { inplace_softmax_k<T>(inputs[x].tensor(), beta); }

 private:
  T beta;
};

template <typename T>
class SoftmaxOperator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct softmax on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 public:
  SoftmaxOperator() : beta(1) {}
  SoftmaxOperator(T beta) : beta(beta) {}
 protected:
  virtual void compute() {
    const Tensor& inT = inputs[in].tensor();
    Tensor& outT = outputs[out].tensor();
    // TODO Check sizes here and throw mismatch
    uint32_t in_size = inT->get_shape().get_linear_size();
    uint32_t out_size = outT->get_shape().get_linear_size();
    if (in_size != out_size)
      Context::get_default_context()->throwError(
          new OperatorIOSizeMismatchError);
    softmax_k<T>(outT, inT, beta);
  }

 private:
  T beta;
};

template <typename T>
class InPlaceSigmoid : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct Sigmoid on non-signed types");

 protected:
  virtual void compute() { inplace_softmax_k<T>(inputs[x].tensor()); }
};

template <typename T>
class SigmoidOperator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct Sigmoid on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    const Tensor& inT = inputs[in].tensor();
    Tensor& outT = outputs[out].tensor();
    // TODO Check sizes here and throw mismatch
    uint32_t in_size = inT->get_shape().get_linear_size();
    uint32_t out_size = outT->get_shape().get_linear_size();
    if (in_size != out_size)
      Context::get_default_context()->throwError(
          new OperatorIOSizeMismatchError);
    sigmoid_k<T>(outT, inT);
  }
};

}  // namespace uTensor

#endif
