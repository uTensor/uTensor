cmake_minimum_required(VERSION 3.10)

project(utensor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)
# TODO be smarter about code coverage bits
# https://github.com/codecov/example-cpp11-cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-rtti")
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -ggdb -fPIC")
endif()

option(ENABLE_PRINTING "Enable debug printing" OFF)
if(ENABLE_PRINTING)
  add_definitions(-DENABLE_PRINTING)
endif()

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

if(ARM_PROJECT)
include_directories(extern/CMSIS_5)
endif()

include_directories(src)
add_subdirectory(src)
# Testing only available if this is the main app # Emergency override MODERN_CMAKE_BUILD_TESTING provided as well i
option(PACKAGE_TESTS "Build the tests" OFF)
option(PACKAGE_TUTORIALS "Build the tutorials" OFF)
#if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING) AND BUILD_TESTING)
#  #add_subdirectory(TESTS) 
#  set(PACKAGE_TESTS ON)
#endif() 
if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(TESTS)
endif()
if(PACKAGE_TUTORIALS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  add_subdirectory(tutorials/error_handling)
  add_subdirectory(tutorials/custom_operator)
endif()

add_subdirectory(tanh_model)
