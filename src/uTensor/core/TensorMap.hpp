#ifndef UTENSOR_TENSOR_MAP
#define UTENSOR_TENSOR_MAP

// #include <algorithm>
// https://github.com/mike-matera/ArduinoSTL/issues/15
#include <initializer_list>

#include "tensor.hpp"
#include "uTensor_util.hpp"
#include "utensor_string.hpp"

using std::initializer_list;

namespace uTensor {
// Tensor maps are fixed size to force input output mismatched errors
class TensorMapInterface {
 public:
  virtual SimpleNamedTensor& operator[](const uTensor::string& name) = 0;
  virtual const SimpleNamedTensor& operator[](
      const uTensor::string& name) const = 0;
  virtual bool has(const uTensor::string& name) const = 0;
  /*
  virtual SimpleNamedTensor& operator[](uint8_t i) = 0;
  virtual const SimpleNamedTensor& operator[](uint8_t i) const = 0;
  */

 public:
  static SimpleNamedTensor not_found;
};

bool compare_named_tensors(const SimpleNamedTensor& a,
                           const SimpleNamedTensor& b);
template <size_t size>
class FixedTensorMap : public TensorMapInterface {
 public:
  FixedTensorMap(SimpleNamedTensor map[size]) : _map{map} {}
  FixedTensorMap(initializer_list<SimpleNamedTensor> l) {
    if (size != l.size()) {
      // TODO THROW ERROR
      uTensor_printf("[Warning] Element number mismatch in TensorMap construction\n");
    }
    for (auto thing = l.begin(); thing != l.end(); thing++) {
      _map[thing->name->get_value()] = *thing;
    }
    //std::sort(std::begin(_map), std::end(_map), compare_named_tensors);
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
    if (!has(name)) {
      return TensorMapInterface::not_found;
    }
    return _map[name.get_value()];
  }
  virtual const SimpleNamedTensor& operator[](
      const uTensor::string& name) const override {
    if (!has(name)){
      return TensorMapInterface::not_found;
    }
    return _map[name.get_value()];
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
    for (size_t i = 0; i < size; i++) _map[i] = that._map[i];
    return *this;
  }
  virtual bool has(const uTensor::string& name) const override {
    const SimpleNamedTensor& x = _map[name.get_value()];
    return x.name != nullptr;
  }
 private:
  SimpleNamedTensor _map[size];
};
}  // namespace uTensor
#endif
