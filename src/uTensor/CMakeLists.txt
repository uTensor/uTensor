add_subdirectory(core)

set(src_utensor_alloc
   )

set(hdr_utensor_alloc
    allocators/arenaAllocator.hpp
   )
    
add_library(utensor ${src_utensor_alloc} ${hdr_utensor_alloc})
target_link_libraries(utensor utensor_core)
target_include_directories(utensor PUBLIC allocators core)
target_compile_features(utensor PUBLIC cxx_std_11)
