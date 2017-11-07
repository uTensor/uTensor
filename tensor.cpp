#include "tensor.hpp"

uint16_t uTensor::incrRef() {
  if(!static_ref_flag) {
    ref_count += 1;
  }

  return ref_count;
}

uint16_t uTensor::dcrRef() {
  ref_count -= 1;
  return ref_count;
}

uint16_t uTensor::getRef() {
  return ref_count;
}

bool uTensor::is_static_ref(void) {
  return static_ref_flag;
}

void uTensor::setStaticRef(uint16_t c) {
  if(ref_count == 0) {
    ref_count = c;
    static_ref_flag = true;
  } else {
    ERR_EXIT("None-zero ref_count");
  }
}



uTensor::~uTensor() {}
