#ifndef UTENSOR_UPAL_H
#define UTENSOR_UPAL_H
#include "uTensor_util.hpp"

// TODO: use macros to register platforms and bit operators to combine flags
// TODO: use bitwise operators for multi-target check, e.g. `#if (UTENSOR_PLATFORM_ARDUINO & __AVR__) == UTENSOR_PLATFORM`

#define UT_PLATFORM(pltform) \
    UT_PLATFORM_CODE == (pltform)

#define UT_PLATFORM_MBED 0
#ifdef MBED
#define UT_PLATFORM_CODE 0
#endif

#define UT_PLATFORM_ARDUINO 1
#ifdef ARDUINO
#define UT_PLATFORM_CODE 1
#endif

#define UT_PLATFORM_X86 2
#ifdef X86
#define UT_PLATFORM_CODE 2
#endif

#if defined(ARDUINO) || defined(MBED_PROJECT)
#define EMBEDDED_PROJECT
#else
#define OS_PROJECT
#endif

#endif
