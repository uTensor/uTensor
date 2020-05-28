#ifndef UTENSOR_ROM_TENSOR_H
#define UTENSOR_ROM_TENSOR_H
#include "BufferTensor.hpp"
#include "context.hpp"

namespace uTensor {

// Literally same behavior as buffer tensor except writing is prohibited
class RomTensor : public BufferTensor {
 protected:
  virtual void* write(uint32_t linear_index) override;

 public:
  RomTensor(TensorShape _shape, ttype _type, const void* buffer);

  // Doing constructors this way lets us check for bounds
  template <typename T, size_t buffer_size>
  RomTensor(TensorShape _shape, const T (&buffer)[buffer_size])
      : BufferTensor(_shape, ttype_from<T>::type,
                     const_cast<void*>(reinterpret_cast<const void*>(buffer))) {
    if (_shape.get_linear_size() != buffer_size) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
  }

  virtual ~RomTensor();
  virtual void resize(TensorShape new_shape) override;

 protected:
  virtual size_t _get_readable_block(const void*& buffer, uint16_t req_read_size,
                                     uint32_t linear_index) const override;
  virtual size_t _get_writeable_block(void*& buffer, uint16_t req_write_size,
                                      uint32_t linear_index) override;
};

class DiagonalRomTensor : public RomTensor {
 protected:
  virtual void* read(uint32_t linear_index) const override;
  virtual void* write(uint32_t linear_index) override;

 public:
  DiagonalRomTensor(TensorShape _shape, ttype _type, const void* buffer,
                    size_t buffer_len);
  virtual ~DiagonalRomTensor();
};
}  // namespace uTensor
#endif
