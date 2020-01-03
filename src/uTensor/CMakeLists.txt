add_subdirectory(core)

set(src_utensor_alloc
   )

set(hdr_utensor_alloc
    allocators/arenaAllocator.hpp
   )
    
set(src_utensor_tensors
  tensors/RamTensor.cpp
  tensors/BufferTensor.cpp
  tensors/RomTensor.cpp
   )
set(hdr_utensor_tensors
  tensors/RamTensor.hpp
  tensors/BufferTensor.hpp
  tensors/RomTensor.hpp
   )
set(src_utensor_ops
   )
set(hdr_utensor_ops
    ops/Arithmetic.hpp
   )
add_library(utensor ${src_utensor_alloc} ${hdr_utensor_alloc} ${src_utensor_tensors} ${hdr_utensor_tensors} ${src_utensor_ops} ${hdr_utensor_ops})
target_link_libraries(utensor utensor_core)
target_include_directories(utensor PUBLIC allocators core tensors ops)
target_compile_features(utensor PUBLIC cxx_std_11)
