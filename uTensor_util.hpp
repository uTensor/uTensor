#ifndef UTENSOR_UTIL
#define UTENSOR_UTIL
#include <stdint.h>
#include <stdio.h>
#include <vector>

// #define MAX(A, B) ((A > B)? A:B)

void return_error(int ret_val);
#if MBED_CONF_APP_DEBUG_MSG

// void errno_error(void* ret_val) {
//   if (ret_val == NULL) {
//     printf(" [**Failure**] %d \r\n", errno);
//     printf("Exiting...\r\n");
//     fflush(stdout);
//     exit(-1);
//   } else {
//     printf("  [DONE]\r\n");
//   }
// }

#define ON_ERR(FUNC, MSG) \
  {                       \
    printf(" * ");        \
    printf(MSG);          \
    return_error(FUNC);   \
  }

#define DEBUG(MSG, ...)         \
  {                             \
    printf(MSG, ##__VA_ARGS__); \
    fflush(stdout);             \
  }

#else  // MBED_CONF_APP_DEBUG_MSG

// void errno_error(void* ret_val) { /*DOES NOTHING*/
// }

#define ON_ERR(FUNC, MSG) FUNC
#define DEBUG(MSG, ...)

#endif

#define ERR_EXIT(MSG, ...)                                      \
  {                                                             \
    printf("[Error] %s:%d @%s ", __FILE__, __LINE__, __func__); \
    printf(MSG, ##__VA_ARGS__);                                 \
    fflush(stdout);                                             \
    exit(-1);                                                   \
  }

typedef std::vector<uint32_t> Shape;

void printVector(std::vector<uint32_t> vec);

// little endian to big endian
uint32_t htonl(uint32_t& val);

// big endian to little endian
uint16_t ntoh16(uint16_t val);
uint32_t ntoh32(uint32_t val);

#endif
