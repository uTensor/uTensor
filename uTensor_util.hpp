#ifndef UTENSOR_UTIL
#define UTENSOR_UTIL

// #define MAX(A, B) ((A > B)? A:B)

#if MBED_CONF_APP_DEBUG_MSG
void return_error(int ret_val){
    if (ret_val) {
      printf(" [**Failure**] %d\r\n", ret_val);
      printf("Exiting...\r\n");
      exit(-1);
    } else {
      printf("  [DONE]\r\n");
    }
}

void errno_error(void* ret_val){
    if (ret_val == NULL) {
      printf(" [**Failure**] %d \r\n", errno);
      error("Exiting...\r\n");
    } else {
      printf("  [DONE]\r\n");
    }
}


#define ON_ERR(FUNC, MSG)   printf(" * "); \
                            printf(MSG); \
                            return_error(FUNC);

#define ASSERT(MSG, ...)    printf(MSG, ##__VA_ARGS__);

#else 

void errno_error(void* ret_val) { /*DOES NOTHING*/}
#define ON_ERR(FUNC, MSG) FUNC
#define ASSERT(MSG, ...)

#endif

void printVector(vector<uint32_t> vec) {
    printf("vector: \r\n");
    for(uint32_t i:vec) {
        printf("%d ", (unsigned int) i);
    }

    printf("\r\n");

}


//little endian to big endian
uint32_t htonl(uint32_t &val) {
    const uint32_t mask = 0b11111111;
    uint32_t ret = 0;
    
    ret |= val >> 24;
    ret |= (val & (mask << 16)) >> 8;
    ret |= (val & (mask << 8)) << 8;
    ret |= val << 24;

    return ret;
}

//big endian to little endian
uint16_t ntoh16(uint16_t val) {
    uint16_t ret = 0;
    
    ret |= val >> 8;
    ret |= val << 8;

    return ret;
}

uint32_t ntoh32(uint32_t val) {
    const uint32_t mask = 0b11111111;
    uint32_t ret = 0;
    
    ret |= val >> 24;
    ret |= (val & (mask << 16)) >> 8;
    ret |= (val & (mask << 8)) << 8;
    ret |= val << 24;

    return ret;
}

#endif