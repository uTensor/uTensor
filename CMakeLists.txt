cmake_minimum_required(VERSION 3.0.2)

project(utensor)

set( CMAKE_VERBOSE_MAKEFILE on )

# set(CMAKE_CXX_FLAGS "-fPIC")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${PROJECT_SOURCE_DIR}/uTensor/core)
include_directories(${PROJECT_SOURCE_DIR}/uTensor/loaders)
include_directories(${PROJECT_SOURCE_DIR}/uTensor/ops)
include_directories(${PROJECT_SOURCE_DIR}/uTensor/util)


file(GLOB SOURCES uTensor/*/*.cpp)
file(GLOB INCLUDES uTensor/*/*.hpp)


message(STATUS "SOURCES: " ${SOURCES})

# target_compile_features(utensor PRIVATE cxx_range_for)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


add_library(utensor STATIC ${SOURCES})

# Configure GTest
enable_testing()
find_package(GTest)
if(GTEST_FOUND)
    find_package(Threads)
    include_directories(${GTEST_INCLUDE_DIRS})
    
    file(GLOB TEST_SOURCES tests/*.cpp)
    add_executable(uTensorTests ${TEST_SOURCES})
    target_link_libraries(uTensorTests utensor ${CMAKE_THREAD_LIBS_INIT})
    GTEST_ADD_TESTS(uTensorTests ${TEST_SOURCES})
endif()
