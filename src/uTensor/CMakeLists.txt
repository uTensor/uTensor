add_subdirectory(core)

option(X86_PROJECT "Build for x86" ON)

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
    ops/Convolution.hpp
    ops/Matrix.hpp
   )
if(X86_PROJECT)
add_definitions( -DX86_PROJECT=1 )
#list(APPEND src_utensor_ops )
endif()
add_subdirectory(ops/optimized)

add_library(utensor ${src_utensor_alloc} ${hdr_utensor_alloc} ${src_utensor_tensors} ${hdr_utensor_tensors} ${src_utensor_ops} ${hdr_utensor_ops} contexts/default_context.cpp)
target_link_libraries(utensor utensor_core optimized_ops)
target_include_directories(utensor PUBLIC allocators core tensors ops)
target_compile_features(utensor PUBLIC cxx_std_11)
