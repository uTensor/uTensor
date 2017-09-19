#include "mbed.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include "tensor.hpp"
#include <stdlib.h>
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include <errno.h>
#include "tensor.hpp"
#include <test.hpp>

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
      printf("Exiting...\r\n");
      exit(-1);
    } else {
      printf("  [DONE]\r\n");
    }
}


#define ON_ERR(FUNC, MSG)   printf(" * "); \
                            printf(MSG); \
                            return_error(FUNC);

                        

// uint32_t hton_f2int(float host_val) {
//     uint32_t tmp = *((uint32_t*) &host_val);
//     return htonl(tmp);
// }

// float ntoh_int2f(uint32_t net_val) {
//     uint32_t tmp = ntoh32(net_val);
//     return *((float *) &tmp);
// }


uint32_t getMagicNumber(unsigned char dtype, unsigned char dim) {
    uint32_t magic = 0;

    magic = (magic | dtype) << 8;
    magic = magic | dim;

    return magic;
}

void printVector(vector<uint32_t> vec) {
    printf("vector: \r\n");
    for(uint32_t i:vec) {
        printf("%d ", (unsigned int) i);
    }

    printf("\r\n");

}

class TensorIdxImporter {
    private:
        FILE *fp;
        HeaderMeta header;
        HeaderMeta parseHeader(void);
        template<typename U>
        TensorBase<U> loader(string &filename, int idx_type);
        void open(string filename);
        //void open(FILE *fp);
        
    public:
        TensorBase<unsigned char> ubyte_import(string filename) { return loader<unsigned char>(filename, idx_ubyte);}
        TensorBase<char> byte_import(string filename)           { return loader<char>(filename, idx_byte);}
        TensorBase<short> short_import(string filename)         { return loader<short>(filename, idx_short);}
        TensorBase<int> int_import(string filename)             { return loader<int>(filename, idx_int);}
        TensorBase<float> float_import(string filename)         { return loader<float>(filename, idx_float);}
        //TensorBase<double> double_import(string filename) {};
};

// void TensorIdxImporter::open(FILE *_fp) {
//     fp = _fp;
//     header = parseHeader();
// }

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
        uint32_t dimSize = ntoh32(*(uint32_t*) buf);
        header.dim.push_back(dimSize);
    }

    free(buf);

    header.dataPos = ftell(fp);
    
    return header;
}

template<typename U>
TensorBase<U> TensorIdxImporter::loader(string &filename, int idx_type) {
    fp = fopen (filename.c_str(), "r" );
    errno_error(fp);

    header = parseHeader();

    fseek(fp, header.dataPos, SEEK_SET);  //need error  handling

    if(header.dataType != idx_type) {
        printf("TensorIdxImporter: header and tensor type mismatch\r\n");
        exit(-1);
    }

    TensorBase<U> t = TensorBase<U>(header.dim);  //tensor allocated
    const uint8_t unit_size = t.unit_size();
    U* val = (U *) malloc(unit_size);
    U* data = t.getPointer({});

    for(uint32_t i = 0; i < t.getSize(); i++) {
        fread(val, unit_size, 1, fp);

        switch (unit_size) {
            case 2:
                *(uint16_t *) val = ntoh16(*(uint16_t *) val);
                break;
            case 4:
                *(uint32_t *) val = ntoh32(*(uint32_t *) val);
                break;
            default:
                break;
        }

        //val = htonl((uint32_t) buff);  //NT: testing for uint8 only, deference error here
        data[i] = *val ;
    }

    free(val);

    ON_ERR(fclose(fp), "Closing file...");

    return t;
}

// Serial pc(USBTX, USBRX, 115200);
// SDBlockDevice bd(D11, D12, D13, D10);
// // SDBlockDevice bd(PTE3, PTE1, PTE2, PTE4);
// //SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
// FATFileSystem fs("fs");

// int main(int argc, char** argv) {
//     while(1) {
//         printf("hello world\r\n");
//         wait(1);
//     }
//     return 0;
// }

class idxImporterTest : public Test {

    private:
        template<typename U>
        double sum(TensorBase<U> input) {
            U* elem = input.getPointer({});
            double accm = 0.0;
            for(uint32_t i = 0; i < input.getSize(); i++) {
                accm += (double) elem[i];
            }
        
            return accm;
        }

    public:

        void ntoh32Test(void) {
            testStart("ntoh32 test");
            uint32_t input = 63;
            uint32_t result = ntoh32(63);
            // printf("ntoh32Test : \r\n");
            // printf("input : ");
            // printBits(sizeof(uint32_t), &input);
            // printf("result : ");
            // printBits(sizeof(uint32_t), &result);
            // printf("\r\n");
            // printf("\r\n");
            passed(result == 1056964608);
        }

        void ucharTest(void) {
            testStart("uchar import test");
            TensorIdxImporter t_import;
            TensorBase<unsigned char> t = t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx");
            double result = sum(t);
            passed(result == 4518);
        }

        void shortTest(void) {
            testStart("short import test");
            TensorIdxImporter t_import;
            TensorBase<short> t = t_import.short_import("/fs/testData/idxImport/int16_4d_power2.idx");
            double result = sum(t);
            passed(result == 270250);
        }

        void intTest(void) {
            testStart("int import test");
            TensorIdxImporter t_import;
            TensorBase<int> t = t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
            double result = sum(t);
            passed(result == 5748992600);
        }

        void floatTest(void) {
            testStart("float import test");
            TensorIdxImporter t_import;
            TensorBase<float> t = t_import.float_import("/fs/testData/idxImport/float_4d_power2.idx");
            
            // printf("printing float ref: \r\n\r\n");

            // auto shape = t.getShape();
            // double tmp = 1.0;
            
            // for(uint16_t i0 = 0; i0 < shape[0]; i0++) {
            //     for(uint16_t i1 = 0; i1 < shape[1]; i1++) {
            //         for(uint16_t i2 = 0; i2 < shape[2]; i2++) {
            //             for(uint16_t i3 = 0; i3 < shape[3]; i3++) {
            //                 tmp = tmp * -0.5;
            //                 printf("%.8e   ", tmp);
            //             }
            //         }
            //     }
            //     tmp = 1.0;
            // }

            // printf("\r\n\r\n");

            // printf("printing float imported: \r\n\r\n");
            // for(uint16_t i0 = 0; i0 < shape[0]; i0++) {
            //     for(uint16_t i1 = 0; i1 < shape[1]; i1++) {
            //         for(uint16_t i2 = 0; i2 < shape[2]; i2++) {
            //             for(uint16_t i3 = 0; i3 < shape[3]; i3++) {
            //                 auto val = t.getPointer({i0,i1,i2,i3});
            //                 printf("%.8e   ", *val);
            //             }
            //         }
            //     }
            // }

            // printf("\r\n\r\n");

            // printf("print binary: ");
            // printBits(sizeof(float), t.getPointer({1,1,1,1}));

            // printf("\r\n\r\n");


            // printf("printing float diff: \r\n\r\n");
            // tmp = 1.0;
            // for(uint16_t i0 = 0; i0 < shape[0]; i0++) {
            //     for(uint16_t i1 = 0; i1 < shape[1]; i1++) {
            //         for(uint16_t i2 = 0; i2 < shape[2]; i2++) {
            //             for(uint16_t i3 = 0; i3 < shape[3]; i3++) {
            //                 tmp = tmp * -0.5;
            //                 auto val = t.getPointer({i0,i1,i2,i3});
            //                 float diff = *val - (float) tmp;
            //                 printf("%.8e   ", diff);
            //             }
            //         }
            //     }
            //     tmp = 1.0;
            // }
            
            // printf("\r\n\r\n");

            double result = sum(t);

            printf("***floating point test yielded: %.8e\r\n", (float) result);
            passed((float)result == -1.0f);
        }

        void runAll(void) {
            ntoh32Test();
            ucharTest();
            shortTest();
            intTest();
            floatTest();
        }

};
