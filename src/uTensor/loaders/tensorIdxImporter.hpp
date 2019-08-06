#ifndef UTENSOR_IDX_IMPORTER
#define UTENSOR_IDX_IMPORTER

#include "src/uTensor/util/uTensor_util.hpp"
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/sdtensor.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <memory>

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
  TensorShape dim;
  long int dataPos;
};

class TensorIdxImporter {
 private:
  FILE* fp;
  HeaderMeta header;
  HeaderMeta parseHeader(void);
  template <typename U>
  Tensor* loader(const char* filename, IDX_DTYPE idx_type);
  template <typename U>
  Tensor* sdloader(const char* filename, IDX_DTYPE idx_type, uint32_t cachesize);
  void parseMeta(const char* filename, IDX_DTYPE idx_type);
  template <typename U>
  void load_impl(U* dst, uint8_t unit_size, uint32_t offset, uint32_t arrsize);
  template <typename U>
  void flush_impl(U* dst, uint8_t unit_size, uint32_t arrsize);
  void open(const char* filename);
  // void open(FILE *fp);

 public:
  Tensor* ubyte_import(const char* filename) {
    return loader<unsigned char>(filename, IDX_DTYPE::idx_ubyte);
  }
  Tensor* byte_import(const char* filename) {
    return loader<char>(filename, IDX_DTYPE::idx_byte);
  }
  Tensor* short_import(const char* filename) {
    return loader<short>(filename, IDX_DTYPE::idx_short);
  }
  Tensor* int_import(const char* filename) {
    return loader<int>(filename, IDX_DTYPE::idx_int);
  }
  Tensor* float_import(const char* filename) {
    return loader<float>(filename, IDX_DTYPE::idx_float);
  }
  Tensor* sd_ubyte_import(const char* filename, uint32_t cachesize) {
    return sdloader<unsigned char>(filename, IDX_DTYPE::idx_ubyte, cachesize);
  }
  Tensor* sd_byte_import(const char* filename, uint32_t cachesize) {
    return sdloader<char>(filename, IDX_DTYPE::idx_byte, cachesize);
  }
  Tensor* sd_short_import(const char* filename, uint32_t cachesize) {
    return sdloader<short>(filename, IDX_DTYPE::idx_short, cachesize);
  }
  Tensor* sd_int_import(const char* filename, uint32_t cachesize) {
    return sdloader<int>(filename, IDX_DTYPE::idx_int, cachesize);
  }
  Tensor* sd_float_import(const char* filename, uint32_t cachesize) {
    return sdloader<float>(filename, IDX_DTYPE::idx_float, cachesize);
  }
  uint32_t getMagicNumber(unsigned char dtype, unsigned char dim);
  template <typename T>
  std::vector<uint32_t> load_data(const char* filename, IDX_DTYPE idx_type, uint8_t unit_size, uint32_t cachesize, uint32_t arrsize, long int offset, T* data);
  template <typename T>
  void flush_data(const char* filename, IDX_DTYPE idx_type, uint8_t unit_size, uint32_t cachesize, uint32_t arrsize, long int offset, T* data);
  uint8_t getIdxDTypeSize(IDX_DTYPE dtype);

  template<typename U>
  void exportFile(const char* filename, IDX_DTYPE idx_type, SDTensor<U>* t, uint8_t unit_size, uint32_t arrsize);
  ~TensorIdxImporter() {
  }
  TensorIdxImporter() {
    fp = NULL;
  }
  // Tensor<double> double_import(string filename) {};
};

// void TensorIdxImporter::open(FILE *_fp) {
//     fp = _fp;
//     header = parseHeader();
// }


template<typename U>
void TensorIdxImporter::exportFile(const char* filename, IDX_DTYPE idx_type, SDTensor<U>* t, uint8_t unit_size, uint32_t arrsize) {
  vm mem = t->getVM();
  FILE* buffer = mem.getFile();
  U* val = (U*)malloc(unit_size);
  parseMeta(filename, idx_type);
  fseek(fp, header.dataPos, SEEK_SET);
  fseek(buffer, 0, SEEK_SET);
  for (uint32_t i = 0; i < arrsize; i++) {
    fread(val, unit_size, 1, fp);
    switch (unit_size) {
      case 2:
        *(uint16_t*)val = ntoh16(*(uint16_t*)val);
        break;
      case 4:
        *(uint32_t*)val = ntoh32(*(uint32_t*)val);
        break;
      default:
        break;
    }
    size_t ret = fwrite(val, unit_size, 1, buffer);
    if (ret != 1) {
        printf("write failed");
        exit(-1);
    }
    fflush(buffer);
  }
  free(val);
  fclose(buffer);
  t->initCache();
}

template <typename U>
void TensorIdxImporter::load_impl(U* dst, uint8_t unit_size, uint32_t offset, uint32_t arrsize) {
  U* val = (U*)malloc(unit_size);
  uint32_t size = offset + arrsize;
  for (uint32_t i = 0; i < size; i++) {
    fread(val, unit_size, 1, fp);
    if (i >= offset) {
    switch (unit_size) {
      case 2:
        *(uint16_t*)val = ntoh16(*(uint16_t*)val);
        break;
      case 4:
        *(uint32_t*)val = ntoh32(*(uint32_t*)val);
        break;
      default:
        break;
    }

    // val = htonl((uint32_t) buff);  //NT: testing for uint8 only, deference
    // error here
   dst[i - offset] = *val;
    }
  }
  free(val);
}
template <typename U>
void TensorIdxImporter::flush_impl(U* res, uint8_t unit_size, uint32_t arrsize) {
  U* dst = (U*)malloc(unit_size * arrsize);
  memcpy(dst, res, (size_t) (unit_size * arrsize));
  for (uint32_t i = 0; i < arrsize; i++) {
    U* val = dst + i;
    switch (unit_size) {
      case 2:
        *(uint16_t*)val = ntoh16(*(uint16_t*)val);
        break;
      case 4:
        *(uint32_t*)val = ntoh32(*(uint32_t*)val);
        break;
      default:
        break;
    }

    // val = htonl((uint32_t) buff);  //NT: testing for uint8 only, deference
    // error here
  }
  fwrite(dst, unit_size, arrsize, fp);
  fflush(fp);
  free(dst);
}
template <typename U>
Tensor* TensorIdxImporter::loader(const char* filename, IDX_DTYPE idx_type) {

  parseMeta(filename, idx_type);
  fseek(fp, header.dataPos, SEEK_SET);
  Tensor* t = new RamTensor<U>(header.dim);  // tensor allocated
  const uint8_t unit_size = t->unit_size();

  U* data = t->write<U>(0, 0);
  load_impl(data, unit_size, 0, t->getSize());

  ON_ERR(fclose(fp), "Closing file...");
  fp = NULL;

  return t;
}

template <typename U>
Tensor* TensorIdxImporter::sdloader(const char* filename, IDX_DTYPE idx_type, uint32_t cachesize) {

  parseMeta(filename, idx_type);
  fseek(fp, header.dataPos, SEEK_SET);
  SDTensor<U>* t = new SDTensor<U>(header.dim, cachesize);  // tensor allocated
  const uint8_t unit_size = t->unit_size();
  uint32_t totalsize = t->getSize();
  exportFile<U>(filename, idx_type, t, unit_size, totalsize);
  ON_ERR(fclose(fp), "Closing file...");

  Tensor* ret = (Tensor*)t;
  return ret;
}

template<typename T>
std::vector<uint32_t> TensorIdxImporter::load_data(const char* filename, IDX_DTYPE idx_type, uint8_t unit_size, uint32_t cachesize, uint32_t arrsize, long int offset, T* data) {
  parseMeta(filename, idx_type);
  if (arrsize == 0) {
    for (auto i : header.dim) {
      if (arrsize == 0) {
        arrsize = i;
      } else {
        arrsize *= i;
      }
    }
  
  }
  int size = std::min(cachesize, arrsize);
  uint32_t offset_t = (uint32_t)offset;
  
  fseek(fp, header.dataPos, SEEK_SET);  // need error  handling
  load_impl(data, unit_size, offset_t, size);

  ON_ERR(fclose(fp), "Closing file...");
  return header.dim;
}
template<typename T>
void TensorIdxImporter::flush_data(const char* filename, IDX_DTYPE idx_type, uint8_t unit_size, uint32_t cachesize, uint32_t arrsize, long int offset, T* data) {
  parseMeta(filename, idx_type);
  int size = std::min(cachesize, arrsize);
  fseek(fp, header.dataPos + offset * unit_size, SEEK_SET);
  flush_impl(data, unit_size, size);
  ON_ERR(fclose(fp), "Closing file...");
  return;
}

#endif  // UTENSOR_IDX_IMPORTER
