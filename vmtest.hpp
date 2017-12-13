#ifndef UTENSOR_VM_TESTS
#define UTENSOR_VM_TESTS

#include "vm.hpp"
#include "test.hpp"
#include "tensorIdxImporter.hpp"

class vmTest : public Test {
  public:
    TensorIdxImporter t_import;
    void loadTest() {
        testStart("load test");
        Tensor* t = t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx", "uchar1");
        unsigned char* data = t->write<unsigned char>(0, 0);
        string tmp_file = "/fs/testData/tmp.txt";
        vm g;
/*        FILE *buf = fopen(tmp_file.c_str(), "w");
        unsigned char* ss = (unsigned char*)malloc(1);
        *ss = 'x';
        size_t s = fwrite(ss, 1, 1, buf);
        
        fflush(buf);
        fclose(buf)*/
        FILE *buf = g.createFile(tmp_file);  
        uint8_t size = (uint8_t)t->unit_size();
        uint32_t totalsize = t->getSize();
        string origin = "/fs/testData/idxImport/uint8_4d_power2.idx";
        t_import.exportFile<unsigned char>(origin, IDX_DTYPE::idx_ubyte, buf, size, totalsize);
        unsigned char* data_g = (unsigned char*)malloc(t->unit_size() * t->getSize());
        g.load_data<unsigned char>(tmp_file, t->unit_size(), 30, t->getSize(), 0,  data_g);
        uint32_t res_x = 0;
        uint32_t res_y = 0;
        for (unsigned int i = 0; i < 30; i++) {
           res_x += data_g[i];
           res_y += data[i];
        }
        g.load_data<unsigned char>(tmp_file, t->unit_size(), 30, t->getSize(), 30,  data_g);
        for (unsigned int i = 0; i < 30; i++) {
           res_x += data_g[i];
           res_y += data[i + 30];
        }

        passed(res_x == res_y);
        free(data_g);
    }
    void writeTest(void) {
      
        testStart("write test");
        Tensor* t = t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx", "uchar1");
        unsigned char* data = t->write<unsigned char>(0, 0);
        string tmp_file = "/fs/testData/tmp2.txt";
        vm g;
        FILE *buf = g.createFile(tmp_file);  
        uint8_t size = (uint8_t)t->unit_size();
        uint32_t totalsize = t->getSize();
        g.flush_data<unsigned char>(tmp_file, t->unit_size(), 30, t->getSize(), 0,  data);
        g.flush_data<unsigned char>(tmp_file, t->unit_size(), 30, t->getSize(), 30,  data + 30);
        unsigned char* data_g = (unsigned char*)malloc(t->unit_size() * 30);
        g.load_data<unsigned char>(tmp_file, t->unit_size(), 30, t->getSize(), 0,  data_g);
        uint32_t res_x = 0;
        uint32_t res_y = 0;
        for (unsigned int i = 0; i < 30; i++) {
           res_x += data_g[i];
           res_y += data[i];
        }
        g.load_data<unsigned char>(tmp_file, t->unit_size(), 30, t->getSize(), 30,  data_g);
        for (unsigned int i = 0; i < 30; i++) {
           res_x += data_g[i];
           res_y += data[i + 30];
        }
        passed(res_x == res_y);

    }

    void runAll(void) {
      loadTest();
      writeTest();
    }
};

#endif
