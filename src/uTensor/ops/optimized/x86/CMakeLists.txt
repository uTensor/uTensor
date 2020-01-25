set(ENABLE_FLOAT ON CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/fftw-3.3.8" "extern/fftw-3.3.8")
set(hdr_optimized_ops 
    FFT.hpp
   )
set(src_optimized_ops 
    FFT.cpp
   )
add_library(optimized_ops ${src_optimized_ops} ${hdr_optimized_ops})
target_link_libraries(optimized_ops utensor_core fftw3f fftw3)
target_include_directories(optimized_ops PUBLIC . ${PROJECT_SOURCE_DIR}/extern/fftw-3.3.8/api)
target_compile_features(optimized_ops PUBLIC cxx_std_11)
