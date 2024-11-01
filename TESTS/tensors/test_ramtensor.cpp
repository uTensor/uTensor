#include "uTensor.h"
#include "uTensor/errorHandlers/SimpleErrorHandler.hpp"
#include "gtest/gtest.h"

#include <iostream>
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

void setup_context() {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
}

TEST(RAM_Tensor, constructor) {
  // setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
}

TEST(RAM_Tensor, read_write_u8) {
  /// setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2, 2) = (uint8_t)5;
  uint8_t read = r(2, 2);
  EXPECT_EQ(read, 5);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, read_write_u8_multi_tensor) {
  /// setup_context();
  localCircularArenaAllocator<512> meta_allocator;
  localCircularArenaAllocator<512> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r1({10, 10}, u8);
  RamTensor r2({10, 10}, u8);
  RamTensor r3({10, 10}, u8);
  r1(2, 2) = (uint8_t)5;
  r2(2, 2) = (uint8_t)5;
  r3(2, 2) = (uint8_t)r1(2, 2) + (uint8_t)r2(2, 2);
  EXPECT_EQ((uint8_t)r3(2, 2), 10);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, read_write_u8_2x) {
  /// setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2, 2) = (uint8_t)5;
  uint8_t read = r(2, 2);
  EXPECT_EQ(read, 5);
  r(2, 2) = (uint8_t)15;
  EXPECT_EQ((uint8_t)r(2, 2), 15);
}

TEST(RAM_Tensor, read_write_u8_contig) {
  /// setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2, 2) = (uint8_t)5;
  r(3, 2) = (uint8_t)35;
  uint8_t read = r(2, 2);
  EXPECT_EQ(read, 5);
  r(2, 2) = (uint8_t)15;
  EXPECT_EQ((uint8_t)r(2, 2), 15);
  EXPECT_EQ((uint8_t)r(3, 2), 35);
}

TEST(RAM_Tensor, read_write_i8) {
  /// setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, i8);
  r(2, 2) = (int8_t)-5;
  int8_t read = r(2, 2);
  EXPECT_EQ(read, -5);
  cout << "i8 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  cout << "Sizeof RamTensor " << sizeof(r) << endl;
}

TEST(RAM_Tensor, read_write_u16) {
  /// setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u16);
  r(2, 2) = (uint16_t)5;
  r(3, 2) = (uint16_t)15;
  uint16_t read = r(2, 2);
  EXPECT_EQ(read, 5);
  read = r(3, 2);
  EXPECT_EQ(read, 15);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, read_write_i16) {
  /// setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, i16);
  r(2, 2) = (int16_t)5;
  r(3, 2) = (int16_t)-15;
  int16_t read = r(2, 2);
  EXPECT_EQ(read, 5);
  read = r(3, 2);
  EXPECT_EQ(read, -15);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, resize_bigger) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({3, 3}, flt);
  EXPECT_EQ(r.num_elems(), 9);
  r.resize({5, 5});
  EXPECT_EQ(r.num_elems(), 25);
}

TEST(RAM_Tensor, resize_bigger_then_back) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({3, 3}, flt);
  r.resize({5, 5});
  r.resize({3, 3});
  EXPECT_EQ(r.num_elems(), 9);
}

TEST(RAM_Tensor, out_of_memory) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<20 * sizeof(float)> ram_allocator;
  bool is_oom = false;
  mErrHandler.set_onError([&is_oom](Error *err) {
    if (*err == OutOfMemError()) {
      is_oom = true;
    }
  });
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  RamTensor a({5, 2}, flt);
  RamTensor b({5, 5}, flt);
  EXPECT_TRUE(is_oom);
}

TEST(RAM_Tensor, move_to_fit) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<25 * sizeof(float)> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor a({5, 2}, flt);
  RamTensor b({2, 5}, flt);
  const void *ori_ptr = b.get_address();
  a.resize({5, 5});
  const void *new_ptr = b.get_address();
  EXPECT_NE(ori_ptr, new_ptr);
}

TEST(RAM_Tensor, view) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<100 * sizeof(int32_t)> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Tensor a = new RamTensor({10, 10}, i32);
  for (int32_t i = 0; i < 100; i++) {
    a(i) = i;
  }
  Tensor view = a.view({1, 8, 2}, {2, 10, 2});
  int32_t read = view(0, 0);
  EXPECT_EQ(read, 12);
  read = view(2, 1);
  EXPECT_EQ(read, 54);
  a(12) = static_cast<int32_t>(100);
  read = view(0, 0);
  EXPECT_EQ(read, 100);
}