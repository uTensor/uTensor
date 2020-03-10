#ifndef UTENSOR_TENSOR_BASE_H
#define UTENSOR_TENSOR_BASE_H
#include "quantizationPrimitives.hpp"
#include "types.hpp"

namespace uTensor {

/** TensorBase is the low level memory interface into tensors
 * It's only job is to place all tensor metadata (vtables, member data, etc)
 * into a known memory location in the Context class
 */
class TensorBase {
 public:
  TensorBase();
  ~TensorBase();

  // Allocate the tensor metadata on a different heap from the data scratch pads
  // Note: as long as derived classes dont override new and delete, these will
  // get called correctly
  void* operator new(size_t sz);
  void operator delete(void* p);
};

// The public interface for all TensorMem types. This is the public contract for
// users of tensors, handling basic data read/write, sizing, construction,
// shaping, etc.
class TensorInterface : public TensorBase {
  // DO not make these read/write calls public or Michael will smite you
 protected:
  virtual void* read(uint32_t linear_index) const = 0;  // Handle to the data
  virtual void* write(uint32_t linear_index) = 0;

 public:
  ttype get_type() const;
  TensorShape& get_shape();
  const TensorShape& get_shape() const;
  TensorInterface();
  TensorInterface(TensorShape _shape, ttype _type);
  TensorInterface(ttype _type);
  virtual ~TensorInterface();

  // Can access Tensors like
  // mTensor(1) = 5, mTensor(2,2) = 5, etc.
  const IntegralValue operator()(uint16_t i, uint16_t j, uint16_t k = 0,
                                 uint16_t l = 0) const;
  IntegralValue operator()(uint16_t i, uint16_t j, uint16_t k = 0,
                           uint16_t l = 0);
  const IntegralValue operator()(uint32_t linear_index) const;
  IntegralValue operator()(uint32_t linear_index);

  virtual void resize(TensorShape new_shape) = 0;
  TensorInterface& set_quantization_params(const QuantizationParams& params);
  const QuantizationParams& get_quantization_params() const;

 private:
  /** Optimized op interface
   * @param buffer pointer to data block to read from/write to in tensor managed
   * memory
   * @param req_(read,write)_size Requested size of read/write block
   * @param linear_index index to start reading from/writing to
   * @return the size of requested block, note may not be equal to the
   * req_read_size/write_size
   */
  size_t get_readable_block(void* buffer, uint16_t req_read_size,
                            uint32_t linear_index) const;
  size_t get_writeable_block(void* buffer, uint16_t req_write_size,
                             uint32_t linear_index);
  friend class FastOperator;

 protected:
  virtual size_t _get_readable_block(void* buffer, uint16_t req_read_size,
                                     uint32_t linear_index) const;
  virtual size_t _get_writeable_block(void* buffer, uint16_t req_write_size,
                                      uint32_t linear_index);

 protected:
  TensorShape _shape;
  ttype _type;  // Maybe make this const
  uint8_t _type_size;
  QuantizationParams _qnt_params;
};

}  // namespace uTensor
#endif
