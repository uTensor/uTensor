cmake_minimum_required(VERSION 3.10)

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

# Build with code coverage
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -ggdb")

add_library(utensor STATIC ${SOURCES})

# Configure GTest
enable_testing()
find_package(GTest)
if(GTEST_FOUND)
    find_package(Threads)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(unity)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_DONT_DEFINE_FOO=1 -DHAVE_GTEST")
    
    # Add the helper header
    include_directories(${PROJECT_SOURCE_DIR}/TESTS)

    file(GLOB TEST_SOURCES TESTS/*/*/test_*.cpp)
    foreach( testsource ${TEST_SOURCES} )
        #string( REPLACE ".cpp" "" testnameT ${testsource} )
        get_filename_component(testname ${testsource} NAME_WE)
        get_filename_component(testdir ${testsource} DIRECTORY)
        file(GLOB TEST_MODEL_SOURCES ${testdir}/models/*.cpp)

        add_executable( ${testname} ${testsource} ${TEST_MODEL_SOURCES} )
        target_link_libraries(${testname} utensor ${CMAKE_THREAD_LIBS_INIT} GTest::GTest GTest::Main unity)
        #add_test(${testname} ${testname})
        # Need this line to work for more detailed test resultso
        # F ME https://blog.kitware.com/dynamic-google-test-discovery-in-cmake-3-10/
        #GTEST_ADD_TESTS(${testname} "" ${testsource})
        gtest_discover_tests(${testname})
    endforeach( testsource ${TEST_SOURCES} )

    #add_executable(uTensorTests ${TEST_SOURCES} tests/gtest_main.cpp)
    #target_link_libraries(uTensorTests utensor ${CMAKE_THREAD_LIBS_INIT} GTest::GTest GTest::Main unity)
    #GTEST_ADD_TESTS(uTensorTests "" AUTO)
endif()

# Make includes accesible to project cmake builds
target_include_directories(utensor PUBLIC ${CMAKE_CURRENT_LIST_DIR}/uTensor/core)
target_include_directories(utensor PUBLIC ${CMAKE_CURRENT_LIST_DIR}/uTensor/loaders)
target_include_directories(utensor PUBLIC ${CMAKE_CURRENT_LIST_DIR}/uTensor/ops)
target_include_directories(utensor PUBLIC ${CMAKE_CURRENT_LIST_DIR}/uTensor/util)


# Todo add install
set(UTENSOR_INCLUDES ${INCLUDES} PARENT_SCOPE)
