#include "src/uTensor/core/vm.hpp"

FILE* vm::createFile(const char* filename) {
  buffer = fopen(filename, "w");
  return buffer;
}
