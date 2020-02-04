#file(GLOB HEADER_LIST "uTensor/allocators/*.hpp;uTensor/core/*.hpp")
add_subdirectory(uTensor)
target_include_directories(utensor 
  PUBLIC 
    $<INSTALL_INTERFACE:.>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)
#add_library(libutensor 
