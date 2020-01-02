#include "types.hpp"

uint8_t type_size(ttype t) {
  switch(t) {
    case i8:
      return sizeof(int8_t);
    case u8:
      return sizeof(uint8_t);
    case i16:
      return sizeof(int16_t);
    case u16:
      return sizeof(uint16_t);
    case i32:
      return sizeof(int32_t);
    case u32:
      return sizeof(uint32_t);
    case flt:
      return sizeof(float);
    default:
      //TODO print error
      return 0;
  }
}
TensorShape::TensorShape(uint16_t shape) : _num_dims(1) {
  _shape[0] = shape;
  _shape[1] = 0;
  _shape[2] = 0;
  _shape[3] = 0;
}
TensorShape::TensorShape(array<uint16_t, 1> shape) : _num_dims(1) {
  _shape[0] = shape[0];
  _shape[1] = 0;
  _shape[2] = 0;
  _shape[3] = 0;
}
TensorShape::TensorShape(array<uint16_t, 2> shape) : _num_dims(2) {
  _shape[0] = shape[0];
  _shape[1] = shape[1];
  _shape[2] = 0;
  _shape[3] = 0;
}
TensorShape::TensorShape(array<uint16_t, 3> shape) : _num_dims(3) {
  _shape[0] = shape[0];
  _shape[1] = shape[1];
  _shape[2] = shape[2];
  _shape[3] = 0;
}
TensorShape::TensorShape(array<uint16_t, 4> shape) : _num_dims(4) {
  _shape[0] = shape[0];
  _shape[1] = shape[1];
  _shape[2] = shape[2];
  _shape[3] = shape[3];
}
TensorShape::TensorShape(uint16_t shape0, uint16_t shape1) : _num_dims(2) {
  _shape[0] = shape0;
  _shape[1] = shape1;
  _shape[2] = 0;
  _shape[3] = 0;
}
TensorShape::TensorShape(uint16_t shape0, uint16_t shape1, uint16_t shape2) : _num_dims(3) {
  _shape[0] = shape0;
  _shape[1] = shape1;
  _shape[2] = shape2;
  _shape[3] = 0;

}
TensorShape::TensorShape(uint16_t shape0, uint16_t shape1, uint16_t shape2, uint16_t shape3) : _num_dims(4){
  _shape[0] = shape0;
  _shape[1] = shape1;
  _shape[2] = shape2;
  _shape[3] = shape3;

}

uint16_t TensorShape::operator[](int i) const {
  return _shape[i]; /* Do additional checks*/
}
uint16_t& TensorShape::operator[](int i) {
  return _shape[i];
}  // Maybe handle update case
void TensorShape::update_dims() {
  for (int i = 0; i < 4; i++) {
    if (_shape[i] && (i + 1) < _num_dims)
      _num_dims = i + 1;
    else if (!_shape[i] && (i + 1) > _num_dims)
      _num_dims = i + 1;
  }
}
uint16_t TensorShape::get_linear_size() const {
  uint16_t sum = 0;
  for (int i = 0; i < _num_dims; i++) {
    sum *= _shape[i];
  }
  return sum;
}

uint8_t TensorShape::num_dims() const { return _num_dims; }
uint32_t TensorShape::linear_index(uint16_t i, uint16_t j, uint16_t k,
                                   uint16_t l) const {
  // TODO
  uint32_t d1 = _shape[1] > 0 ? 1 : 0;  d1*= _shape[0];
  uint32_t d2 = _shape[2] > 0 ? 1 : 0;  d2*= d1*_shape[1];
  uint32_t d3 = _shape[3] > 0 ? 1 : 0;  d3*= d2*_shape[2];

  return i + j*d1 + k*d2 + l*d3;
}
IntegralValue::IntegralValue(void* p) : p(p) {}
IntegralValue::IntegralValue(const uint8_t& u) {
  *reinterpret_cast<uint8_t*>(p) = u;
}
IntegralValue::IntegralValue(const int8_t& u) {
  *reinterpret_cast<int8_t*>(p) = u;
}
IntegralValue::IntegralValue(const uint16_t& u) {
  *reinterpret_cast<uint16_t*>(p) = u;
}
IntegralValue::IntegralValue(const int16_t& u) {
  *reinterpret_cast<int16_t*>(p) = u;
}
IntegralValue::IntegralValue(const uint32_t& u) {
  *reinterpret_cast<uint32_t*>(p) = u;
}
IntegralValue::IntegralValue(const int32_t& u) {
  *reinterpret_cast<int32_t*>(p) = u;
}

IntegralValue::operator uint8_t() const {
  return static_cast<uint8_t>(*reinterpret_cast<uint8_t*>(p));
}
IntegralValue::operator uint8_t&() {
  return static_cast<uint8_t&>(*reinterpret_cast<uint8_t*>(p));
}
IntegralValue::operator int8_t() const {
  return static_cast<int8_t>(*reinterpret_cast<int8_t*>(p));
}
IntegralValue::operator int8_t&() {
  return static_cast<int8_t&>(*reinterpret_cast<int8_t*>(p));
}
IntegralValue::IntegralValue::operator uint16_t() const {
  return static_cast<uint16_t>(*reinterpret_cast<uint16_t*>(p));
}
IntegralValue::operator uint16_t&() {
  return static_cast<uint16_t&>(*reinterpret_cast<uint16_t*>(p));
}
IntegralValue::operator int16_t() const {
  return static_cast<int16_t>(*reinterpret_cast<int16_t*>(p));
}
IntegralValue::operator int16_t&() {
  return static_cast<int16_t&>(*reinterpret_cast<int16_t*>(p));
}
IntegralValue::IntegralValue::operator uint32_t() const {
  return static_cast<uint32_t>(*reinterpret_cast<uint32_t*>(p));
}
IntegralValue::operator uint32_t&() {
  return static_cast<uint32_t&>(*reinterpret_cast<uint32_t*>(p));
}
IntegralValue::operator int32_t() const {
  return static_cast<int32_t>(*reinterpret_cast<int32_t*>(p));
}
IntegralValue::operator int32_t&() {
  return static_cast<int32_t&>(*reinterpret_cast<int32_t*>(p));
}
IntegralValue::IntegralValue::operator float() const {
  return static_cast<float>(*reinterpret_cast<float*>(p));
}
IntegralValue::operator float&() {
  return static_cast<float&>(*reinterpret_cast<float*>(p));
}
