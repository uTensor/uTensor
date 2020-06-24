#ifndef UTENSOR_BUFFER_TENSOR_H
#define UTENSOR_BUFFER_TENSOR_H

#include "context.hpp"
#include "tensorBase.hpp"

namespace uTensor {
class BufferTensor : public TensorInterface {
 protected:
  virtual void* read(uint32_t linear_index) const override;
  virtual void* write(uint32_t linear_index) override;

 public:
  BufferTensor(const TensorShape& _shape, ttype _type);
  BufferTensor(const TensorShape& _shape, ttype _type, void* buffer);

  // Doing constructors this way lets us check for bounds
  template <typename T, size_t buffer_size>
  BufferTensor(const TensorShape& _shape, T (&buffer)[buffer_size])
      : BufferTensor(_shape, ttype_from<T>::type,
                     reinterpret_cast<void*>(buffer)) {
    if (_shape.get_linear_size() != buffer_size) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
  }

  bool is_bound_to_buffer() const;
  // Check if is bound to this pointer
  bool is_bound_to_buffer(void* b) const;
  bool bind(void* b);
  bool unbind();

  virtual ~BufferTensor();
  // Does nothing
  virtual void resize(const TensorShape& new_shape) override;

 protected:
  uint8_t* _buffer;
};
}  // namespace uTensor
#endif
