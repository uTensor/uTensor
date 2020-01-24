#ifndef UTENSOR_ROM_TENSOR_H
#define UTENSOR_ROM_TENSOR_H
#include "BufferTensor.hpp"

namespace uTensor {

// Literally same behavior as buffer tensor except writing is prohibited
class RomTensor : public BufferTensor {
 protected:
  virtual void* write(uint32_t linear_index) override;

 public:
  RomTensor(TensorShape _shape, ttype _type, const void* buffer);
  virtual ~RomTensor();
  virtual void resize(TensorShape new_shape) override;

 protected:
  virtual size_t _get_readable_block(void* buffer, uint16_t req_read_size,
                                     uint32_t linear_index) const override;
  virtual size_t _get_writeable_block(void* buffer, uint16_t req_write_size,
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
