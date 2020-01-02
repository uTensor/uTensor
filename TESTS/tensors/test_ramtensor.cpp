#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "RamTensor.hpp"

#include <iostream>
using std::cout;
using std::endl;

using namespace uTensor;

void setup_context(){
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);
}

TEST(RAM_Tensor, constructor) {
  //setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
}

TEST(RAM_Tensor, read_write_u8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2,2) = (uint8_t) 5;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, read_write_u8_2) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2,2) = (uint8_t) 5;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  r(2,2) = (uint8_t) 15;
  EXPECT_EQ((uint8_t)r(2,2), 15);
}

TEST(RAM_Tensor, read_write_i8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, i8);
  r(2,2) = (int8_t) -5;
  int8_t read = r(2,2);
  EXPECT_EQ(read, -5);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}
