#include "mbed.h"
#include <stdlib.h>
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include <stdio.h>
#include <errno.h>
#include "tensor.hpp"
#include "tensorIdxImporter.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(D11, D12, D13, D10);
// SDBlockDevice bd(PTE3, PTE1, PTE2, PTE4);
//SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char** argv) {
    printf("test start: \r\n");

    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");
    printf("fs mounted \r\n");

    idxImporterTest test;
    test.runAll();

    printf("Test Summaries:\r\n");
    test.printSummary();

    return 0;
}

// template <class T>
// class X
// {
//     std::vector<T> v;
//     X() = default; //Terminating recursion

//     public:
//     template <class U, class... Ts>
//     X(U n, Ts... rest)  : X(rest...) {
//         printf("%d\n\r", sizeof...(rest));
//     }

// Serial pc(USBTX, USBRX, 115200);
// SDBlockDevice bd(D11, D12, D13, D10);
// FATFileSystem fs("fs");

// void return_error(int ret_val){
//     if (ret_val)
//       printf("Failure. %d\r\n", ret_val);
//     else
//       printf("done.\r\n");
// }


// void errno_error(void* ret_val){
//     if (ret_val == NULL)
//       printf(" Failure. %d \r\n", errno);
//     else
//       printf(" done.\r\n");
//   }
  

// int main(int argc, char** argv) {
//     int error = 0;
//     printf("Mounting the filesystem on \"/fs\". ");
//     error = fs.mount(&bd);
//     return_error(error);

//     printf("out.txt.");
//     FILE* fd = fopen("/fs/out.txt", "w");
//     errno_error(fd);

//     fprintf(fd, "Hello World!");

//     printf("Closing file.");
//     fclose(fd);
//     printf(" done.\r\n");

//     // TensorBase<char> obj({2, 2});

//     // printf("stride dim0: %d, dim0: %d\r\n", obj.getStride(0), obj.getStride(1));

//     // *(obj.getPointer({0,0})) = 1;
//     // *(obj.getPointer({0,1})) = 1;
//     // *(obj.getPointer({1,0})) = 0;
//     // *(obj.getPointer({1,1})) = 0;

//     // char* elem = obj.getPointer({});

//     // printf("%d %d %d %d\r\n", elem[0], elem[1], elem[2], elem[3]);

//     return 0;
// }
