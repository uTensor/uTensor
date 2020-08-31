#ifndef UTENSOR_UPAL_H
#define UTENSOR_UPAL_H
#include "uTensor_util.hpp"
#include <limits>
#include <float.h>

// TODO: use macros to register platforms and bit operators to combine flags
// TODO: use bitwise operators for multi-target check, e.g. `#if (UTENSOR_PLATFORM_ARDUINO & __AVR__) == UTENSOR_PLATFORM`

#define UT_ARCH(arch) \
    (defined(UT_ARCH_CODE) && UT_ARCH_CODE == (arch))

#define UT_PLATFORM(pltform) \
    (defined(UT_PLATFORM_CODE) && (UT_PLATFORM_CODE == (pltform)))

///////// Platform Code

#define UT_PLATFORM_MBED 0
#ifdef MBED
#define UT_PLATFORM_CODE 0
#endif

#define UT_PLATFORM_ARDUINO 1
#ifdef ARDUINO
#define UT_PLATFORM_CODE 1
#endif

//////// Arch Code

#define UT_ARCH_ARM 0
#if !(defined(X86) || defined(__AVR__)) //arch check condition
#define UT_ARCH_CODE 0
#endif


#define UT_ARCH_X86 1
#ifdef X86
#define UT_ARCH_CODE 1
#endif

#define UT_ARCH_AVR 2
#ifdef __AVR__
#define UT_ARCH_CODE 2
#endif



////////
#if defined(ARDUINO) || defined(MBED_PROJECT)
#define EMBEDDED_PROJECT
#else
#define OS_PROJECT
#endif

/////// AVR Arduino Hack
#if UT_ARCH(UT_ARCH_AVR) && UT_PLATFORM(UT_PLATFORM_ARDUINO)
namespace std {
    template<typename _Tp>
    struct is_signed
    {
    static bool const value = _Tp(-1) < _Tp(0);
    };
}
#endif  // AVR Arduino Hack

namespace uTensor {
namespace uPAL {
    template<typename T>
    T lowest() {
        T val;
        if (std::numeric_limits<T>::has_infinity) {
            if((val = -LDBL_MIN) == -LDBL_MIN) return val;
            if((val = -DBL_MIN) == -DBL_MIN) return val;
            if((val = -FLT_MIN) == -FLT_MIN) return val;
        } else {
            return std::numeric_limits<T>::min();
        }
    }
}
}


#endif
