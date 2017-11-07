#include "uTensor_util.hpp"
#include <cstdlib> 

//Doesnt need to be in #define
void return_error(int ret_val) {
  if (ret_val) {
    printf(" [**Failure**] %d\r\n", ret_val);
    printf("Exiting...\r\n");
    fflush(stdout);
    exit(-1);
  } else {
    printf("  [DONE]\r\n");
  }
}
void printVector(std::vector<uint32_t> vec) {
  printf("vector: \r\n");
  for (uint32_t i : vec) {
    printf("%d ", (unsigned int)i);
  }

  printf("\r\n");
}

// little endian to big endian
uint32_t htonl(uint32_t& val) {
  const uint32_t mask = 0b11111111;
  uint32_t ret = 0;

  ret |= val >> 24;
  ret |= (val & (mask << 16)) >> 8;
  ret |= (val & (mask << 8)) << 8;
  ret |= val << 24;

  return ret;
}

// big endian to little endian
uint16_t ntoh16(uint16_t val) {
  uint16_t ret = 0;

  ret |= val >> 8;
  ret |= val << 8;

  return ret;
}

uint32_t ntoh32(uint32_t val) {
  const uint32_t mask = 0b11111111;
  uint32_t ret = 0;

  ret |= val >> 24;
  ret |= (val & (mask << 16)) >> 8;
  ret |= (val & (mask << 8)) << 8;
  ret |= val << 24;

  return ret;
}
