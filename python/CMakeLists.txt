pybind11_add_module(
    _pyuTensor
    cpp/_pyuTensor.cpp
    cpp/impl/fast_copyop.cpp
    cpp/impl/matmul.cpp
    cpp/impl/conv.cpp
    cpp/impl/allocator.cpp
    cpp/impl/arithmetic_kernels.cpp
)
target_include_directories(_pyuTensor PUBLIC cpp/include)
target_link_libraries(_pyuTensor PRIVATE utensor)
set_target_properties(
    _pyuTensor
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/pyuTensor
)