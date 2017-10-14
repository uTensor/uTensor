#ifndef UTENSOR_IDX_IMPORTER
#define UTENSOR_IDX_IMPORTER

#include "mbed.h"
#include <vector>
#include <stdio.h>
#include "tensor.hpp"
#include <stdlib.h>
#include "uTensor_util.hpp"

using namespace std;

enum IDX_DTYPE {
    idx_ubyte = 0x08,
    idx_byte = 0x09,
    idx_short = 0x0B,
    idx_int = 0x0C,
    idx_float = 0x0D,
    idx_double = 0x0E
};

class HeaderMeta {
    public:
        IDX_DTYPE dataType;
        unsigned char numDim;
        vector<uint32_t> dim;
        long int dataPos;
};

class TensorIdxImporter {
    private:
        FILE *fp;
        HeaderMeta header;
        HeaderMeta parseHeader(void);
        template<typename U>
        Tensor<U> loader(string &filename, IDX_DTYPE idx_type);
        void open(string filename);
        //void open(FILE *fp);
        
    public:
        Tensor<unsigned char> ubyte_import(string filename) { return loader<unsigned char>(filename, IDX_DTYPE::idx_ubyte);}
        Tensor<char> byte_import(string filename)           { return loader<char>(filename, IDX_DTYPE::idx_byte);}
        Tensor<short> short_import(string filename)         { return loader<short>(filename, IDX_DTYPE::idx_short);}
        Tensor<int> int_import(string filename)             { return loader<int>(filename, IDX_DTYPE::idx_int);}
        Tensor<float> float_import(string filename)         { return loader<float>(filename, IDX_DTYPE::idx_float);}
        uint32_t getMagicNumber(unsigned char dtype, unsigned char dim);
        uint8_t getIdxDTypeSize(IDX_DTYPE dtype) ;
        //Tensor<double> double_import(string filename) {};
};

// void TensorIdxImporter::open(FILE *_fp) {
//     fp = _fp;
//     header = parseHeader();
// }

uint8_t TensorIdxImporter::getIdxDTypeSize(IDX_DTYPE dtype) {

    switch(dtype) {
        case idx_ubyte:
            return 1;
        case idx_byte:
            return 1;
        case idx_short:
            return 2;
        case idx_int:
            return 4;
        case idx_float:
            return 4;
        case idx_double:
            return 8;
    }

    return 0;
}

uint32_t TensorIdxImporter::getMagicNumber(unsigned char dtype, unsigned char dim) {
    uint32_t magic = 0;

    magic = (magic | dtype) << 8;
    magic = magic | dim;

    return magic;
}

HeaderMeta TensorIdxImporter::parseHeader(void) {
    unsigned char *buf = (unsigned char*) malloc(sizeof(unsigned char) * 4);

    fread(buf, 1, 4, fp);
    if(buf[0] != 0 || buf[0] != 0) {
        printf("Error, header magic number invalid\r\n");
    }

    HeaderMeta header;
    header.dataType = static_cast<IDX_DTYPE>(buf[2]);
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
Tensor<U> TensorIdxImporter::loader(string &filename, IDX_DTYPE idx_type) {
    fp = fopen (filename.c_str(), "r" );

    DEBUG("Opening file %s ", filename.c_str());
    if(fp == NULL) ERR_EXIT("Error opening file: %s", filename.c_str());

    header = parseHeader();

    if(header.dataType != idx_type) {
        ERR_EXIT("TensorIdxImporter: header and tensor type mismatch\r\n");
    }

    fseek(fp, header.dataPos, SEEK_SET);  //need error  handling

    Tensor<U> t = Tensor<U>(header.dim);  //tensor allocated
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

#endif  //UTENSOR_IDX_IMPORTER
