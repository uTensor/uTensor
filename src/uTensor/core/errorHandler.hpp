#ifndef UTENSOR_ERROR_H
#define UTENSOR_ERROR_H

namespace uTensor {

struct Event {};
struct Error : public Event {};

// Simplest possible error handler, Users can roll their own
class ErrorHandler {
 public:
  virtual void uThrow(Error* err);
  virtual void notify(const Event& evt);
};

// Default errors
struct InvalidReshapeError : public Error {};
struct InvalidResizeError : public Error {};
struct InvalidMemAccessError : public Error {};
struct OutOfMemError : public Error {};
struct OutOfMemBoundsError : public Error {};
struct InvalidOptimizableTensorError : public Error {};
struct InvalidTensorError : public Error {};
struct InvalidTensorInputError : public Error {};
struct InvalidTensorOutputError : public Error {};

};  // namespace uTensor
#endif
