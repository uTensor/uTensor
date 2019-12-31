#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include <iostream>
using std::cout;
using std::endl;

using namespace uTensor;
TEST(ArenaAllocator, constructor) {

  localCircularArenaAllocator<256> _allocator;
  EXPECT_GE(_allocator.available(), 240); // F u 64Bit machine self alignment

}

TEST(ArenaAllocator, single_alloc) {

  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(40);
  EXPECT_GE(_allocator.available(), 240 - 40 - 16);
  EXPECT_EQ(_allocator.contains(ptr), true);

}

TEST(ArenaAllocator, oversized_alloc) {

  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(400);
  EXPECT_EQ(ptr, nullptr);

}

TEST(ArenaAllocator, deallocate) {

  localCircularArenaAllocator<256> _allocator;
  void* ptr = nullptr;
  EXPECT_EQ(_allocator.contains(ptr), false);
  ptr = _allocator.allocate(40);
  EXPECT_EQ(_allocator.contains(ptr), true);
  _allocator.deallocate(ptr);
  EXPECT_EQ(_allocator.contains(ptr), false);

}

TEST(ArenaAllocator, single_alloc_access) {

  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(40);
  // Check for seg faults
  uint8_t* data = reinterpret_cast<uint8_t*>(ptr);
  for(int i = 0; i < 40; i++){
    data[i] = i;
  }

}

TEST(ArenaAllocator, two_alloc_access) {

  localCircularArenaAllocator<256> _allocator;
  void* ptr1 = _allocator.allocate(40);
  void* ptr2 = _allocator.allocate(40);
  // Check for seg faults
  uint8_t* data1 = reinterpret_cast<uint8_t*>(ptr1);
  uint8_t* data2 = reinterpret_cast<uint8_t*>(ptr2);
  for(int i = 0; i < 40; i++){
    data1[i] = i;
  }
  for(int i = 0; i < 40; i++){
    data2[i] = 255-i;
  }
  for(int i = 0; i < 40; i++){
    EXPECT_EQ(data1[i], i);
  }
  for(int i = 0; i < 40; i++){
    EXPECT_EQ(data2[i], 255-i);
  }

}

TEST(ArenaAllocator, circle_back) {
  localCircularArenaAllocator<256> _allocator;
  void* ptr1 = _allocator.allocate(100);
  cout << "Avaliable " << _allocator.available() << endl;
  void* ptr2 = _allocator.allocate(100);
  cout << "Avaliable " << _allocator.available() << endl;
  void* ptr3 = _allocator.allocate(117);
  cout << "Avaliable " << _allocator.available() << endl;
  EXPECT_EQ(_allocator.contains(ptr2), false);
  EXPECT_EQ(_allocator.contains(ptr3), true);
  EXPECT_EQ(_allocator.contains(ptr1), true);

}
