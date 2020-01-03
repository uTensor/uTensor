add_subdirectory(core)

set(src_utensor_alloc
   )

set(hdr_utensor_alloc
    allocators/arenaAllocator.hpp
   )
    
set(src_utensor_tensors
  tensors/RamTensor.cpp
  tensors/BufferTensor.cpp
   )
set(hdr_utensor_tensors
  tensors/RamTensor.hpp
  tensors/BufferTensor.hpp
   )
add_library(utensor ${src_utensor_alloc} ${hdr_utensor_alloc} ${src_utensor_tensors} ${hdr_utensor_tensors})
target_link_libraries(utensor utensor_core)
target_include_directories(utensor PUBLIC allocators core tensors)
target_compile_features(utensor PUBLIC cxx_std_11)
