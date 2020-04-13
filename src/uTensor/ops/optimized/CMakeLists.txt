if(X86_PROJECT)
  #add_subdirectory(x86)
  #add_library(optimized_ops 
elseif(ARM_PROJECT)
  set(ROOT ${PROJECT_SOURCE_DIR}/extern/CMSIS_5)
  set(DSP ${ROOT}/CMSIS/DSP)
  list(APPEND CMAKE_MODULE_PATH ${DSP})
  add_subdirectory(arm)
endif()
