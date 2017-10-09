#include "mbed.h"
#include "stdlib.h"
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "stdio.h"
#include "tensor.hpp"
#include "tensorIdxImporter.hpp"
#include "MatrixTest.hpp"
#include "ArrayOps.hpp"
#include "MathOps.hpp"
#include "NnOps.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char** argv) {
    printf("test start: \r\n");
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    //test running..
    printf("running idx import tests...\r\n");
    idxImporterTest idxTest;
    idxTest.runAll();

    printf("running matrix op tests...\r\n");
    matrixOpsTest matrixTests;
    matrixTests.runAll();

    printf("running array op tests...\r\n");
    ArrayOpsTest arrayTests;
    arrayTests.runAll();

    printf("running math op tests...\r\n");
    MathOpsTest mathTests;
    mathTests.runAll();

    printf("running nn op tests...\r\n");
    NnOpsTest nnTests;
    nnTests.runAll();

    //end of test runs

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
    
    //print the results
    printf("========= Test Summaries ========= \r\n");
    printf("========= IDX import:\r\n");
    idxTest.printSummary();
    printf("========= Matrix Ops:\r\n");
    matrixTests.printSummary();
    printf("========= Array Ops:\r\n");
    arrayTests.printSummary();
    printf("========= Math Ops:\r\n");
    mathTests.printSummary();
    printf("========= NN Ops:\r\n");
    nnTests.printSummary();
    printf("==================================\r\n");
    printf("==================================\r\n");

    return 0;
}
