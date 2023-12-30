pybind11_add_module(_pyuTensor _pyuTensor.cpp play_fastop.cpp)
target_link_libraries(_pyuTensor PRIVATE utensor)
