#ifndef UTENSOR_TENSOR_MAP
#define UTENSOR_TENSOR_MAP

#include <initializer_list>
#include <algorithm>
#include "tensor.hpp"
#include "utensor_string.hpp"

using std::initializer_list;

namespace uTensor {
// Tensor maps are fixed size to force input output mismatched errors
class TensorMapInterface {
 public:
  virtual SimpleNamedTensor& operator[](const uTensor::string& name) = 0;
  virtual const SimpleNamedTensor& operator[](
      const uTensor::string& name) const = 0;
  /*
  virtual SimpleNamedTensor& operator[](uint8_t i) = 0;
  virtual const SimpleNamedTensor& operator[](uint8_t i) const = 0;
  */

 public:
  static SimpleNamedTensor not_found;
};

bool compare_named_tensors(const SimpleNamedTensor& a, const SimpleNamedTensor& b);
template <size_t size>
class FixedTensorMap : public TensorMapInterface {
 public:
  FixedTensorMap(SimpleNamedTensor map[size]) : _map{map} {}
  FixedTensorMap(initializer_list<SimpleNamedTensor> l) {
    if (size != l.size()) {
      // TODO THROW ERROR
      printf("Element number mismatch in TensorMap construction\n");
    }
    int i = 0;
    for (auto thing = l.begin(); thing != l.end(); thing++) {
      _map[i] = *thing;
      i++;
    }
    std::sort(std::begin(_map), std::end(_map), compare_named_tensors);
  }
  FixedTensorMap() {
    //_map = {not_found};
  }
  virtual ~FixedTensorMap() {}
  /*
  SimpleNamedTensor& operator[](uint8_t i) override {
    return _map[i];
  }
  virtual const SimpleNamedTensor& operator[](uint8_t i) const override {
    return _map[i];
  }
  */
  virtual SimpleNamedTensor& operator[](const uTensor::string& name) override {
    for (int i = 0; i < size; i++) {
      if (name == *(_map[i].name)) return _map[i];
    }
    return TensorMapInterface::not_found;
  }
  virtual const SimpleNamedTensor& operator[](const uTensor::string& name) const override {
    for (int i = 0; i < size; i++) {
      if (name == *(_map[i].name)) return _map[i];
    }
    return TensorMapInterface::not_found;
  }
  FixedTensorMap(FixedTensorMap<size>&& that) {
    _map = that._map;
    that._map = nullptr;
  }
  FixedTensorMap& operator=(FixedTensorMap<size>&& that) {
    if (this != &that) {
      _map = that._map;
      that._map = nullptr;
    }
    return *this;
  }
  FixedTensorMap(const FixedTensorMap<size>& that) { _map = that._map; }
  FixedTensorMap& operator=(const FixedTensorMap<size>& that) {
    //_map = that._map;
    for (int i = 0; i < size; i++) _map[i] = that._map[i];
    return *this;
  }

 private:
  SimpleNamedTensor _map[size];
};
}  // namespace uTensor
#endif
