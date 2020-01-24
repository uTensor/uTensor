#file(GLOB HEADER_LIST "uTensor/allocators/*.hpp;uTensor/core/*.hpp")
add_subdirectory(uTensor)
target_include_directories(utensor PUBLIC .)
#add_library(libutensor 
