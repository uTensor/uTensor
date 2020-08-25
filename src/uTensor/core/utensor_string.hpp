#ifndef UTENSOR_STRING
#define UTENSOR_STRING
//#include <string.h>
#include <cstdint>
#include <cstring>

namespace uTensor {
class string {
 private:
  uint32_t value;
  const char* cstr = NULL;

  uint32_t hash(const char* c) {
    int v = 7;
    for (uint16_t i = 0; i < strlen(c); i++) {
      v = v * 31 + c[i];
    }

    return (uint32_t)v;
  }

 public:
  string(const char* that) {
    value = hash(that);
    cstr = that;
  }
  string() { value = 0; }
  string(uint32_t val) : value(val), cstr(NULL) {}
  // bool operator < (const string& that){ return this->value < that.value; }
  // bool operator == (const string& that){ return this->value == that.value; }
  bool operator<(const string& that) const { return this->value < that.value; }
  bool operator==(const string& that) const {
    return this->value == that.value;
  }
  bool operator!=(const string& that) const {
    return this->value != that.value;
  }

  uint32_t get_value() const { return value; }
  const char* c_str() const { return cstr; }
};

}  // namespace uTensor

// namespace std {
// template <>
// struct hash<uTensor::string> {
//  typedef uTensor::string argument_type;
//  typedef std::size_t result_type;
//
//  result_type operator()(argument_type const& s) const noexcept {
//    return s.get_value();
//  }
//};
//}  // namespace std

#endif
