#include "uTensor/core/vm.hpp"

FILE* vm::createFile(std::string& filename) {
  buffer = fopen(filename.c_str(), "w");
  return buffer;
}
