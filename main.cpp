#include "mbed.h"
#include <stdlib.h>
#include "tensor.hpp"

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

Serial pc(USBTX, USBRX, 115200);
// LocalFileSystem local("local");

int main(int argc, char** argv) {

    // FILE *fp = fopen("/local/out.txt", "w");  // Open "out.txt" on the local file system for writing
    // fprintf(fp, "Hello World!");
    // printf("hello world\r\n");
    // fclose(fp);

    // TensorBase<char> obj({2, 2});

    // printf("stride dim0: %d, dim0: %d\r\n", obj.getStride(0), obj.getStride(1));

    // *(obj.getPointer({0,0})) = 1;
    // *(obj.getPointer({0,1})) = 1;
    // *(obj.getPointer({1,0})) = 0;
    // *(obj.getPointer({1,1})) = 0;

    // char* elem = obj.getPointer({});

    // printf("%d %d %d %d\r\n", elem[0], elem[1], elem[2], elem[3]);

    return 0;
}
