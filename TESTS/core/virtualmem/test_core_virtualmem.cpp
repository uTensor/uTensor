#include "test_helper.h"
#include "src/uTensor/loaders/tensorIdxImporter.hpp"

#include <iostream>
#include <algorithm>
#include <random>

using std::cout;
using std::endl;

TensorIdxImporter t_import;
Context ctx;

void test_core_vmload() {
        Tensor* t = t_import.ubyte_import("/fs/constants/idxImport/uint8_4d_power2.idx");
        unsigned char* data = t->write<unsigned char>(0, 0);
        string tmp_file = "/fs/constants/sdtmp/tmp.txt";
        vm g;
        unsigned char* data_g = (unsigned char*)malloc(t->getSize_in_bytes());
        g.createFile(tmp_file.c_str());
        g.flush_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 0,  data);
        g.load_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 0,  data_g);
        uint32_t res_x = 0;
        uint32_t res_y = 0;
        for (unsigned int i = 0; i < 30; i++) {
           res_x += data_g[i];
           res_y += data[i];
        }
        g.flush_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 30,  data + 30);
        g.load_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 30,  data_g);
        for (unsigned int i = 0; i < 30; i++) {
           res_x += data_g[i];
           res_y += data[i + 30];
        }

        EXPECT_EQ(res_x, res_y);
        free(data_g);
}

void test_core_vmwrite(void) {
      
        Tensor* t = t_import.ubyte_import("/fs/constants/idxImport/uint8_4d_power2.idx");
        unsigned char* data = t->write<unsigned char>(0, 0);
        string tmp_file = "/fs/constants/sdtmp/tmp2.txt";
        vm g;
        FILE *buf = g.createFile(tmp_file.c_str());  
        uint8_t size = (uint8_t)t->unit_size();
        uint32_t totalsize = t->getSize();
        g.flush_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 0,  data);
        unsigned char* data_g = (unsigned char*)malloc(t->unit_size() * 30);
        g.load_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 0,  data_g);
        uint32_t res_x = 0;
        uint32_t res_y = 0;
        for (unsigned int i = 0; i < 30; i++) {
           res_x += (uint32_t)data_g[i];
           res_y += data[i];
           EXPECT_EQ(res_x, res_y);
        }
        g.flush_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 30,  data + 30);
        g.load_data<unsigned char>(tmp_file.c_str(), t->unit_size(), 30, t->getSize(), 30,  data_g);
        for (unsigned int i = 0; i < 30; i++) {
           res_x += data_g[i];
           res_y += data[i + 30];
        }
        EXPECT_EQ(res_x, res_y);

}

UTENSOR_TEST_CONFIGURE()

UTENSOR_TEST(core, vmload, "virtual memory load test")
UTENSOR_TEST(core, vmwrite, "virtual memory write test")

UTENSOR_TEST_RUN()
