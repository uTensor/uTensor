#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include <iostream>
using std::cout;
using std::endl;

using namespace uTensor;

TEST(Context, default_constructor) {
  EXPECT_EQ(Context::get_metadata_allocator(), nullptr);
  EXPECT_EQ(Context::get_ram_data_allocator(), nullptr);
}

TEST(Context, set_allocators_same_instance) {
  localCircularArenaAllocator<256> _allocator;
  Context::set_metadata_allocator(&_allocator);
  Context::set_ram_data_allocator(&_allocator);
  EXPECT_EQ(Context::get_metadata_allocator(), &_allocator);
  EXPECT_EQ(Context::get_ram_data_allocator(), &_allocator);
}

TEST(Context, set_allocators_same) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);
  EXPECT_EQ(Context::get_metadata_allocator(), &meta_allocator);
  EXPECT_EQ(Context::get_ram_data_allocator(), &ram_allocator);
}
