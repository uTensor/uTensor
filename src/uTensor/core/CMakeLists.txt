set(src_utensor_core
    types.cpp
    uTensor_util.cpp
    tensorBase.cpp
    memoryManagementInterface.cpp
    context.cpp
    tensor.cpp
   )

set(hdr_utensor_core
    types.hpp
    uTensor_util.hpp
    tensorBase.hpp
    memoryManagementInterface.hpp
    context.hpp
    tensor.hpp
   )
    
add_library(libutensor_core ${src_utensor_core} ${hdr_utensor_core})
target_include_directories(libutensor_core PUBLIC .)
target_compile_features(libutensor_core PUBLIC cxx_std_11)
