#ifndef UTENSOR_TENSOR_MAP
#define UTENSOR_TENSOR_MAP

// Tensor maps are fixed size to force input output mismatched errors
class TensorMapInterface {
 public:
  virtual SimpleNamedTensor& operator[](const utensor::string& name) = 0;
  virtual const SimpleNamedTensor& operator[](
      const utensor::string& name) const = 0;

 public:
  static SimpleNamedTensor not_found(utensor::string("NotFound"),
                                     static_cast<utensor::Tensor>(NULL));
};

template <size_t size>
class FixedTensorMap : public TensorMapInterface {
 public:
  TensorMap(SimpleNamedTensor map[size]) : _map[map] {}
  virtual ~TensorMap() {}
  SimpleNamedTensor& operator[](const utensor::string& name) {
    for (int i = 0; i < size; i++) {
      if (name == _map[i].name) return _map[i];
    }
    return TensorMapInterface::not_found;
  }
  const SimpleNamedTensor& operator[](const utensor::string& name) const {
    for (int i = 0; i < size; i++) {
      if (name == _map[i].name) return _map[i];
    }
    return TensorMapInterface::not_found;
  }

 private:
  SimpleNamedTensor _map[size];
  FixedTensorMap(const FixedTensorMap& that) {}
};
#endif
