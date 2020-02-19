#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include "SimpleErrorHandler.hpp"
#include <iostream>
#include <algorithm>

using std::cout;
using std::endl;

using namespace uTensor;
TEST(ArenaAllocator, constructor) {

  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  localCircularArenaAllocator<256> _allocator;
  int d = 1;

  EXPECT_GE(_allocator.available(), 240); // F u 64Bit machine self alignment
  EXPECT_EQ(*errH.begin(), localCircularArenaAllocatorConstructed());

}

TEST(ArenaAllocator, single_alloc) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);

  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(40);
  EXPECT_EQ(_allocator.available(), 256 - 40);
  EXPECT_EQ(_allocator.contains(ptr), true);

}

TEST(ArenaAllocator, oversized_alloc) {
  SimpleErrorHandler errH(50);
  errH.set_onError([](Error* err) { EXPECT_EQ(*err, OutOfMemError()); } );
  Context::get_default_context()->set_ErrorHandler(&errH);

  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(400);
  EXPECT_EQ(ptr, nullptr);

}

TEST(ArenaAllocator, deallocate) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);

  localCircularArenaAllocator<256> _allocator;
  void* ptr = nullptr;
  EXPECT_EQ(_allocator.contains(ptr), false);
  ptr = _allocator.allocate(40);
  EXPECT_EQ(_allocator.contains(ptr), true);
  _allocator.deallocate(ptr);
  EXPECT_EQ(_allocator.contains(ptr), false);

}

TEST(ArenaAllocator, single_alloc_access) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);

  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(40);
  // Check for seg faults
  uint8_t* data = reinterpret_cast<uint8_t*>(ptr);
  for(int i = 0; i < 40; i++){
    data[i] = i;
  }

}

TEST(ArenaAllocator, two_alloc_access) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);

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

// Less of a test, more of a code coverage forcing hand
TEST(ArenaAllocator, alloc_reuse) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);

  localCircularArenaAllocator<256> _allocator;
  void* ptr1 = _allocator.allocate(40);
  uint8_t* data1 = reinterpret_cast<uint8_t*>(ptr1);
  for(int i = 0; i < 40; i++){
    data1[i] = i;
  }
  _allocator.deallocate(ptr1);

  void* ptr2 = _allocator.allocate(40);
  uint8_t* data2 = reinterpret_cast<uint8_t*>(ptr2);
  for(int i = 0; i < 40; i++){
    data2[i] = 255-i;
  }
  for(int i = 0; i < 40; i++){
    EXPECT_EQ(data2[i], 255-i);
  }

}

/**
 * Unbound pointers are not guaranteed to be valid if we overflow for whatever reason
 */
TEST(ArenaAllocator, circle_back) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  localCircularArenaAllocator<256> _allocator;

  void* ptr1 = _allocator.allocate(100);
  for(int i = 0 ; i < 100; i++){
    reinterpret_cast<uint8_t*>(ptr1)[i] = 0x01;
  }
  cout << "Avaliable " << _allocator.available() << endl;
  void* ptr2 = _allocator.allocate(100);
  for(int i = 0 ; i < 100; i++){
    reinterpret_cast<uint8_t*>(ptr2)[i] = 0x02;
  }
  cout << "Avaliable " << _allocator.available() << endl;
  void* ptr3 = _allocator.allocate(117);
  for(int i = 0 ; i < 117; i++){
    reinterpret_cast<uint8_t*>(ptr3)[i] = 0x03;
  }
  cout << "Avaliable " << _allocator.available() << endl;
  EXPECT_EQ(_allocator.contains(ptr2), false);
  EXPECT_EQ(_allocator.contains(ptr3), true);
  //EXPECT_EQ(_allocator.contains(ptr1), true); Ptr1 is invalidated but basically just ptr3 in this mem manageer
  //Check to make sure data is correct
  for(int i = 0 ; i < 100; i++){
    EXPECT_NE(reinterpret_cast<uint8_t*>(ptr1)[i], 0x01);
  }
  for(int i = 0 ; i < 117; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(ptr3)[i], 0x03);
  }

  // Check to make sure a rebalance has occurred
  bool has_rebalanced = std::find(errH.begin(), errH.end(), localCircularArenaAllocatorRebalancing()) != errH.end();
  EXPECT_EQ(has_rebalanced, true);
}

/**
 * Regions bound to a handle can be moved around inside the memory management system without worry
 */
TEST(ArenaAllocator, circle_back_with_handle) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  localCircularArenaAllocator<256> _allocator;
  void* ptr1 = _allocator.allocate(96);
  for(int i = 0 ; i < 100; i++){
    reinterpret_cast<uint8_t*>(ptr1)[i] = 0x01;
  }
  cout << "Avaliable " << _allocator.available() << endl;
  void* ptr2 = _allocator.allocate(100);
  Handle hndl2(ptr2);
  bind(hndl2, _allocator);
  for(int i = 0 ; i < 100; i++){
    reinterpret_cast<uint8_t*>(ptr2)[i] = 0x02;
  }
  // This will deallocate the first bit
  cout << "Avaliable " << _allocator.available() << endl;
  void* ptr3 = _allocator.allocate(117);
  for(int i = 0 ; i < 117; i++){
    reinterpret_cast<uint8_t*>(ptr3)[i] = 0x03;
  }
  cout << "Avaliable " << _allocator.available() << endl;
  // We moved ptr2 in the allocator rebalance, it should be gone
  EXPECT_EQ(_allocator.contains(ptr2), false);
  EXPECT_EQ(_allocator.contains(*hndl2), true);
  EXPECT_EQ(_allocator.contains(ptr3), true);
  //EXPECT_EQ(_allocator.contains(ptr1), true); Ptr1 is invalidated but basically just ptr3 in this mem manageer
  //Check to make sure data is correct
  for(int i = 0 ; i < 100; i++){
    EXPECT_NE(reinterpret_cast<uint8_t*>(ptr1)[i], 0x01);
  }
  // Dereference the handle instead
  for(int i = 0 ; i < 100; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(*hndl2)[i], 0x02);
  }
  for(int i = 0 ; i < 117; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(ptr3)[i], 0x03);
  }
  // Check to make sure a rebalance has occurred
  bool has_rebalanced = std::find(errH.begin(), errH.end(), localCircularArenaAllocatorRebalancing()) != errH.end();
  EXPECT_EQ(has_rebalanced, true);
}

TEST(Handle, default_construction) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  Handle h;
  EXPECT_EQ(sizeof(Handle), sizeof(void*));
}

TEST(Handle, construction) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(100);
  Handle hndl(ptr);
}

TEST(Handle, read_write) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(100);
  Handle hndl(ptr);
  for(int i = 0 ; i < 100; i++){
    reinterpret_cast<uint8_t*>(*hndl)[i] = 0x01;
    EXPECT_EQ(reinterpret_cast<uint8_t*>(*hndl)[i], 0x01);
  }
}

TEST(HandleReference, default_construction) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  Handle h;
  EXPECT_EQ(sizeof(Handle), sizeof(void*));
}

TEST(HandleReference, copy) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(100);
  Handle hndl(ptr);
  for(int i = 0 ; i < 100; i++){
    reinterpret_cast<uint8_t*>(*hndl)[i] = 0x01;
  }
  HandleReference h2 = hndl;
  reinterpret_cast<uint8_t*>(*h2)[0] = 0x02;
  EXPECT_EQ(reinterpret_cast<uint8_t*>(*hndl)[0], 0x02);
  for(int i = 1 ; i < 100; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(*hndl)[i], 0x01);
  }
  for(int i = 1 ; i < 100; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(*h2)[i], 0x01);
  }

}

TEST(HandleReference, copy_copy) {
  SimpleErrorHandler errH(50);
  Context::get_default_context()->set_ErrorHandler(&errH);
  localCircularArenaAllocator<256> _allocator;
  void* ptr = _allocator.allocate(100);
  Handle hndl(ptr);
  for(int i = 0 ; i < 100; i++){
    reinterpret_cast<uint8_t*>(*hndl)[i] = 0x01;
  }
  HandleReference h1 = hndl;
  HandleReference h2 = h1;
  reinterpret_cast<uint8_t*>(*h2)[0] = 0x02;
  EXPECT_EQ(reinterpret_cast<uint8_t*>(*hndl)[0], 0x02);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(*h1)[0], 0x02);
  for(int i = 1 ; i < 100; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(*hndl)[i], 0x01);
  }
  for(int i = 1 ; i < 100; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(*h1)[i], 0x01);
  }
  for(int i = 1 ; i < 100; i++){
    EXPECT_EQ(reinterpret_cast<uint8_t*>(*h2)[i], 0x01);
  }

}
