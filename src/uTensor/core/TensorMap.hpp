#ifndef UTENSOR_TENSOR_MAP
#define UTENSOR_TENSOR_MAP

#include "utensor_string.hpp"
#include "tensor.hpp"
#include <initializer_list>

using std::initializer_list;

namespace uTensor {
// Tensor maps are fixed size to force input output mismatched errors
class TensorMapInterface {
 public:
  virtual SimpleNamedTensor& operator[](const uTensor::string& name) = 0;
  virtual const SimpleNamedTensor& operator[](
      const uTensor::string& name) const = 0;

 public:
  static SimpleNamedTensor not_found;
};

template <size_t size>
class FixedTensorMap : public TensorMapInterface {
 public:
  FixedTensorMap(SimpleNamedTensor map[size]) : _map{map} {}
  FixedTensorMap(initializer_list<SimpleNamedTensor> l) {
    if(size != l.size()){
        //TODO THROW ERROR
        printf("Element number mismatch in TensorMap construction\n");
    }
    int i = 0;
    for(auto thing = l.begin(); thing != l.end(); thing++){
        _map[i] = *thing;
        i++;
    }
  }
  FixedTensorMap() {
    //_map = {not_found};
  }
  virtual ~FixedTensorMap() {}
  SimpleNamedTensor& operator[](const uTensor::string& name) {
    for (int i = 0; i < size; i++) {
      if (name == *(_map[i].name)) return _map[i];
    }
    return TensorMapInterface::not_found;
  }
  const SimpleNamedTensor& operator[](const uTensor::string& name) const {
    for (int i = 0; i < size; i++) {
      if (name == *(_map[i].name)) return _map[i];
    }
    return TensorMapInterface::not_found;
  }
  FixedTensorMap(FixedTensorMap<size>&& that){
      _map = that._map;
      that._map = nullptr;
  } 
  FixedTensorMap& operator=(FixedTensorMap<size>&& that) {
    if(this != &that){
      _map = that._map;
      that._map = nullptr;
    }
      return *this;
  }
  FixedTensorMap(const FixedTensorMap<size>& that) {
    _map = that._map;
  }
  FixedTensorMap& operator=(const FixedTensorMap<size>& that) {
    //_map = that._map;
    for(int i = 0; i < size; i++)
        _map[i] = that._map[i];
    return *this;
  }

 private:
  SimpleNamedTensor _map[size];
};
}
#endif
