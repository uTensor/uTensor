#include "test.hpp"

void printBits(size_t const size, void const* const ptr) {
  unsigned char* b = (unsigned char*)ptr;
  unsigned char byte;
  int i, j;

  for (i = size - 1; i >= 0; i--) {
    for (j = 7; j >= 0; j--) {
      byte = (b[i] >> j) & 1;
      printf("%d", byte);
    }
  }
  puts("");
}

void printFloatData(Tensor* tensor) {
  float v;
  const float* ptr_data = tensor->read<float>(0, 0);
  for (size_t i = 0; i < tensor->getSize(); ++i) {
    v = *(ptr_data + i);
    printf("%f, ", v);
  }
  printf("\r\n");
  return;
}
