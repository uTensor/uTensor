# Borrowed and modified from CMSIS DSP
# Setting Linux is forcing th extension to be .o instead of .obj when building on WIndows.
# It is important because armlink is failing when files have .obj extensions (error with
# scatter file section not found)
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)



#SET(tools "C:/PROGRA~2/GNUTOO~1/82018-~1")

#SET(CMAKE_C_COMPILER "${tools}/bin/arm-none-eabi-gcc${EXE}")
#SET(CMAKE_CXX_COMPILER "${tools}/bin/arm-none-eabi-g++${EXE}")
#SET(CMAKE_ASM_COMPILER "${tools}/bin/arm-none-eabi-gcc${EXE}")

find_program(CMAKE_C_COMPILER NAMES arm-none-eabi-gcc arm-none-eabi-gcc${EXE})
find_program(CMAKE_CXX_COMPILER NAMES arm-none-eabi-g++ arm-none-eabi-g++${EXE})
find_program(CMAKE_ASM_COMPILER NAMES arm-none-eabi-gcc arm-none-eabi-gcc${EXE})

#SET(CMAKE_AR "${tools}/bin/arm-none-eabi-gcc-ar${EXE}")
find_program(CMAKE_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar${EXE})
find_program(CMAKE_CXX_COMPILER_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar${EXE})
find_program(CMAKE_C_COMPILER_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar${EXE})


#SET(CMAKE_CXX_COMPILER_AR "${tools}/bin/arm-none-eabi-gcc-ar${EXE}")
#SET(CMAKE_C_COMPILER_AR "${tools}/bin/arm-none-eabi-gcc-ar${EXE}")

#SET(CMAKE_LINKER "${tools}/bin/arm-none-eabi-g++${EXE}")
find_program(CMAKE_LINKER NAMES arm-none-eabi-g++ arm-none-eabi-g++${EXE})

SET(CMAKE_C_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
SET(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
SET(CMAKE_C_OUTPUT_EXTENSION .o)
SET(CMAKE_CXX_OUTPUT_EXTENSION .o)
SET(CMAKE_ASM_OUTPUT_EXTENSION .o)
# When library defined as STATIC, this line is needed to describe how the .a file must be
# create. Some changes to the line may be needed.
SET(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> -crs <TARGET> <LINK_FLAGS> <OBJECTS>" )
SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> -crs <TARGET> <LINK_FLAGS> <OBJECTS>" )

set(GCC ON)
# default core

if(NOT ARM_CPU)
    set(
            ARM_CPU "cortex-a5"
            CACHE STRING "Set ARM CPU. Default : cortex-a5"
    )
endif(NOT ARM_CPU)

SET(CMAKE_C_FLAGS "-g -ffunction-sections -fdata-sections -mcpu=${ARM_CPU}" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_CXX_FLAGS "-g -ffunction-sections -fdata-sections -mcpu=${ARM_CPU}" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_ASM_FLAGS "-mcpu=${ARM_CPU}" CACHE INTERNAL "ASM compiler common flags")
#SET(CMAKE_EXE_LINKER_FLAGS "--specs=nosys.specs"  CACHE INTERNAL "linker flags")

get_property(IS_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE)
if(IS_IN_TRY_COMPILE)
    add_link_options("--specs=nosys.specs")
endif()

add_link_options("-Wl,--start-group")

# Where is the target environment
#SET(CMAKE_FIND_ROOT_PATH "${tools}")
# Search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# For libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

