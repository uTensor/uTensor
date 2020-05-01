add_subdirectory(core)

option(X86_PROJECT "Build for x86" OFF)
option(ARM_PROJECT "Build for Arm" OFF)

set(src_utensor_alloc
    allocators/arenaAllocator.cpp
   )

set(src_utensor_tensors
  tensors/RamTensor.cpp
  tensors/BufferTensor.cpp
  tensors/RomTensor.cpp
   )
set(src_utensor_ops
    ops/Matrix.cpp
   )
set(src_utensor_errhndl
  errorHandlers/SimpleErrorHandler.cpp
   )

set(src_utensor_legacy_ops
    ops/legacy/LegacyMath.cpp
    ops/legacy/legacyQuantizationUtils.cpp
    )
set(src_utensor_sq_ops
    ops/symmetric_quantization/depthwise_separable_convolution.cpp
    ops/symmetric_quantization/depthwise_separable_convolution_kernels.cpp
    ops/symmetric_quantization/symmetric_quantization_utils.cpp
   )

if(X86_PROJECT)
add_definitions( -DX86_PROJECT=1 )
elseif(ARM_PROJECT)
add_definitions( -DARM_PROJECT=1 )
#list(APPEND src_utensor_ops )
endif()
add_subdirectory(ops/optimized)

add_library(utensor  ${src_utensor_alloc} ${src_utensor_tensors} ${src_utensor_ops} ${src_utensor_errhndl} contexts/default_context.cpp ${src_utensor_legacy_ops} ${src_utensor_sq_ops} )
#target_link_libraries(utensor utensor_core optimized_ops)
target_link_libraries(utensor PUBLIC utensor_core )
target_include_directories(utensor PUBLIC allocators tensors ops ops/legacy ops/symmetric_quantization errorHandlers)
#target_include_directories(utensor PUBLIC 
#  $<BUILD_INTERFACE: allocators tensors ops>
#  $<INSTALL_INTERFACE: ${CMAKE_CURRENT_SOURCE_DIR}/allocators ${CMAKE_CURRENT_SOURCE_DIR}/tensors ${CMAKE_CURRENT_SOURCE_DIR}/ops>
#  )
target_compile_features(utensor PUBLIC cxx_std_11)
