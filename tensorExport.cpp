#include "mbed.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include "tensor.hpp"

using namespace std;

#define idx_ubyte 0x08
#define idx_byte  0x09
#define idx_short 0x0B
#define idx_int   0x0C
#define idx_float 0x0D
#define idx_double 0x0E

class HeaderMeta {
    public:
        unsigned char dataType;
        unsigned char numDim;
        vector<uint32_t> dim;
        long int dataPos;
};

//little endian to big endian
uint32_t htonl(uint32_t &val) {
    // const uint32_t mask = 0b11111111;
    uint32_t ret = 0;
    
    ret |= val >> 24;
    ret |= (val << 8) >> 16;
    ret |= (val << 16) >> 8;
    ret |= val << 24;

    // ret |= (val & (mask << 24)) >> 24;
    // ret |= (val & (mask << 16)) >> 8;
    // ret |= (val & (mask << 8)) << 8;
    // ret |= (val & (mask << 0)) << 24;

    return ret;
}

//big endian to little endian
uint32_t ntohl(uint32_t &val) {
    uint32_t ret = 0;
    
    ret |= val >> 24;
    ret |= (val << 8) >> 16;
    ret |= (val << 16) >> 8;
    ret |= val << 24;

    return ret;
}

uint32_t hton_f2int(float host_val) {
    uint32_t tmp = *((uint32_t*) &host_val);
    return htonl(tmp);
}

float ntoh_int2f(uint32_t net_val) {
    uint32_t tmp = ntohl(net_val);
    return *((float *) &tmp);
}

class TensorIdxImporter {
    private:
        FILE *fp;
        HeaderMeta header;
        HeaderMeta parseHeader(void);
        void idxTensor8bitHelper(vector<uint32_t> trace, vector<uint32_t>::iterator it);
        TensorBase<unsigned char> t;
        
    public:
        TensorIdxImporter(string filename);
        TensorIdxImporter(FILE *fp);
        TensorBase<unsigned char> idxTensor8bit(void);
};

TensorIdxImporter::TensorIdxImporter(string filename) {
    fp = fopen (filename.c_str(), "r" );
    if (fp==NULL) {
        printf("Error opening file\r\n");
        exit(-1);
    }

    header = parseHeader();
}

TensorIdxImporter::TensorIdxImporter(FILE *_fp) {
    fp = _fp;
    header = parseHeader();
}

uint32_t getMagicNumber(unsigned char dtype, unsigned char dim) {
    uint32_t magic = 0;

    magic = (magic | dtype) << 8;
    magic = magic | dim;

    return magic;
}

void printVector(vector<uint32_t> vec) {
    printf("vector: \r\n");
    for(uint32_t i:vec) {
        printf("%d ", i);
    }

    printf("\r\n");

}

HeaderMeta TensorIdxImporter::parseHeader(void) {
    unsigned char *buf = (unsigned char*) malloc(sizeof(unsigned char) * 4);

    fread(buf, 1, 4, fp);
    if(buf[0] != 0 || buf[0] != 0) {
        printf("Error, header magic number invalid\r\n");
    }

    HeaderMeta header;
    header.dataType = buf[2];
    header.numDim = buf[3];

    for(int i = 0; i < header.numDim; i++) {
        fread(buf, 1, 4, fp);
        uint32_t dimSize = ntohl(*(uint32_t*) buf);
        header.dim.push_back(dimSize);
    }

    free(buf);

    header.dataPos = ftell(fp);
    
    return header;
}

void TensorIdxImporter::idxTensor8bitHelper(vector<uint32_t> trace, vector<uint32_t>::iterator it) {
    if(it+1 != header.dim.end()) {
        for(uint32_t i = 0; i < *it; i++) {
            auto branch = trace;
            branch.push_back(i);
            // printf("%d ", i);
            idxTensor8bitHelper(branch, it+1);
        }
        
        // printf("\r\n");

    } else {    //last dimension
        unsigned char* dp = t.getPointer(trace);
        for(uint32_t i = 0; i < *it; i++) {
            int val = fgetc(fp);
            if(val == EOF) {
                printf("EoF reached\r\n");
                exit(-1);
            }
            dp[i] = (unsigned char) val;
        }
        printVector(trace);
    }

}

TensorBase<unsigned char> TensorIdxImporter::idxTensor8bit(void) {
    if(header.dataType != idx_ubyte) {
        printf("The IDX file doesn't contain an unsigned-char tensor\r\n");
        exit(-1);
    }

    fseek(fp, header.dataPos, SEEK_SET);
    t = TensorBase<unsigned char>(header.dim);  //tensor allocated
    vector<uint32_t> trace;
    idxTensor8bitHelper(trace, header.dim.begin());

    return t;
}

// Serial pc(USBTX, USBRX, 115200);

// int main(int argc, char** argv) {

//     TensorIdxImporter t_import("./idx/uint8_4d_power2.idx");
//     TensorBase<unsigned char> t = t_import.idxTensor8bit();

//     //TensorBase<unsigned char> t = idxTensor8bit(header, fp);
//     printf("something\r\n");
//     unsigned char* elem = t.getPointer({});

//     printf("size: %d\r\n", (int) t.getSize());
//     printf("data\r\n");
//     for(int i = 0; i < t.getSize(); i++) {
//         printf("%d ", elem[i]);
//     }
//     printf("\r\n");

//     return 0;
// }
// //csv write
// //or THE IDX FILE FORMAT