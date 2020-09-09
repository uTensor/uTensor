if(X86_PROJECT)
  #add_subdirectory(x86)
  #add_library(optimized_ops 
elseif(ARM_PROJECT)
  add_subdirectory(arm)
endif()
