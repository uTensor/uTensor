#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include "uTensor/core/context.hpp"
#include "BufferTensor.hpp"

#include <iostream>
using std::cout;
using std::endl;

using namespace uTensor;

void setup_context(){
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
}

TEST(Buffer_Tensor, constructor) {
  //setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t* buffer = new uint8_t[10*10];
  BufferTensor r({10, 10}, u8, buffer);
  delete[] buffer;
}

TEST(Buffer_Tensor, safe_constructor) {
  //setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t buffer[10*10];
  BufferTensor r({10, 10}, buffer);
}

TEST(Buffer_Tensor, read_write_u8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t* buffer = new uint8_t[10*10];
  BufferTensor r({10, 10}, u8, buffer);
  r(2,2) = (uint8_t) 5;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  delete[] buffer;
}

TEST(Buffer_Tensor, read_write_u8_2x) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t* buffer = new uint8_t[10*10];
  BufferTensor r({10, 10}, u8, buffer);
  r(2,2) = (uint8_t) 5;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  r(2,2) = (uint8_t) 15;
  EXPECT_EQ((uint8_t)r(2,2), 15);
  delete[]  buffer;
}

TEST(Buffer_Tensor, read_write_u8_contig) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t* buffer = new uint8_t[10*10];
  BufferTensor r({10, 10}, u8, buffer);
  r(2,2) = (uint8_t) 5;
  r(3,2) = (uint8_t) 35;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  r(2,2) = (uint8_t) 15;
  EXPECT_EQ((uint8_t)r(2,2), 15);
  EXPECT_EQ((uint8_t)r(3,2), 35);
  delete[] buffer;
}

TEST(Buffer_Tensor, read_write_i8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  int8_t* buffer = new int8_t[10*10];
  BufferTensor r({10, 10}, i8, buffer);
  r(2,2) = (int8_t) -5;
  int8_t read = r(2,2);
  EXPECT_EQ(read, -5);
  cout << "i8 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  cout << "Sizeof BufferTensor " << sizeof(r) << endl;
  delete[] buffer;
}

TEST(Buffer_Tensor, read_write_u16) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint16_t* buffer = new uint16_t[10*10];
  BufferTensor r({10, 10}, u16, buffer);
  r(2,2) = (uint16_t) 5;
  r(3,2) = (uint16_t) 15;
  uint16_t read = r(2,2);
  EXPECT_EQ(read, 5);
  read = r(3,2);
  EXPECT_EQ(read, 15);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  delete[] buffer;
}

TEST(Buffer_Tensor, read_write_i16) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  int16_t* buffer = new int16_t[10*10];
  BufferTensor r({10, 10}, i16, buffer);
  r(2,2) = (int16_t) 5;
  r(3,2) = (int16_t) -15;
  int16_t read = r(2,2);
  EXPECT_EQ(read, 5);
  read = r(3,2);
  EXPECT_EQ(read, -15);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  delete[] buffer;
}
