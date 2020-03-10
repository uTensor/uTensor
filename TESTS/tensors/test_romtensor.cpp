#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "RomTensor.hpp"

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

TEST(Rom_Tensor, constructor) {
  //setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t* buffer = new uint8_t[10*10];
  for(int i = 0; i < 100; i++)
      buffer[i] = i;
  const RomTensor r({10, 10}, u8, buffer);
  delete[] buffer;
}

TEST(Rom_Tensor, fixed_constructor) {
  //setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t buffer[10*10];
  for(int i = 0; i < 100; i++)
      buffer[i] = i;
  const RomTensor r({10, 10}, buffer);
}

TEST(Rom_Tensor, read_write_u8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t* buffer = new uint8_t[10*10];
  for(int i = 0; i < 100; i++)
      buffer[i] = i;
  const RomTensor r({10, 10}, u8, buffer);
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 22);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  delete[] buffer;
}

TEST(Rom_Tensor, read_write_u8_2x) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint8_t* buffer = new uint8_t[10*10];
  for(int i = 0; i < 100; i++)
      buffer[i] = i;
  const RomTensor r({10, 10}, u8, buffer);
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 22);
  EXPECT_EQ((uint8_t)r(3,2), 32);
  delete[]  buffer;
}

TEST(Rom_Tensor, read_write_i8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  int8_t* buffer = new int8_t[10*10];
  for(int i = 0; i < 100; i++)
      buffer[i] = i;
  const RomTensor r({10, 10}, i8, buffer);
  int8_t read = r(2,2);
  EXPECT_EQ(read, 22);
  cout << "i8 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  cout << "Sizeof RomTensor " << sizeof(r) << endl;
  delete[] buffer;
}

TEST(Rom_Tensor, read_write_u16) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  uint16_t* buffer = new uint16_t[10*10];
  for(int i = 0; i < 100; i++)
      buffer[i] = i;
  const RomTensor r({10, 10}, u16, buffer);
  uint16_t read = r(2,2);
  EXPECT_EQ(read, 22);
  read = r(3,2);
  EXPECT_EQ(read, 32);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  delete[] buffer;
}

TEST(Rom_Tensor, read_write_i16) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  int16_t* buffer = new int16_t[10*10];
  for(int i = 0; i < 100; i++)
      buffer[i] = i;
  const RomTensor r({10, 10}, i16, buffer);
  int16_t read = r(2,2);
  EXPECT_EQ(read, 22);
  read = r(3,2);
  EXPECT_EQ(read, 32);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  delete[] buffer;
}
