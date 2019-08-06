#include "src/uTensor/loaders/tensorIdxImporter.hpp"

uint8_t TensorIdxImporter::getIdxDTypeSize(IDX_DTYPE dtype) {
  switch (dtype) {
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

uint32_t TensorIdxImporter::getMagicNumber(unsigned char dtype,
                                           unsigned char dim) {
  uint32_t magic = 0;

  magic = (magic | dtype) << 8;
  magic = magic | dim;

  return magic;
}

HeaderMeta TensorIdxImporter::parseHeader(void) {
  unsigned char* buf = (unsigned char*)malloc(sizeof(unsigned char) * 4);

  fread(buf, 1, 4, fp);
  if (buf[0] != 0 || buf[0] != 0) {
    printf("Error, header magic number invalid\r\n");
  }

  HeaderMeta header;
  header.dataType = static_cast<IDX_DTYPE>(buf[2]);
  header.numDim = buf[3];

  for (int i = 0; i < header.numDim; i++) {
    fread(buf, 1, 4, fp);
    uint32_t dimSize = ntoh32(*(uint32_t*)buf);
    header.dim.push_back(dimSize);
  }

  free(buf);

  header.dataPos = ftell(fp);

  return header;
}

void TensorIdxImporter::parseMeta(const char* filename, IDX_DTYPE idx_type) {
  fp = fopen(filename, "r");

  DEBUG("Opening file %s ", filename);
  if (fp == NULL) ERR_EXIT("Error opening file: %s", filename);

  header = parseHeader();

  if (header.dataType != idx_type) {
    ERR_EXIT("TensorIdxImporter: header and tensor type mismatch\r\n");
  }


}
