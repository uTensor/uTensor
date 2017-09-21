#include "mbed.h"
#include "stdlib.h"
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "stdio.h"
#include "tensor.hpp"
#include "tensorIdxImporter.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char** argv) {
    printf("test start: \r\n");

    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    idxImporterTest test;
    test.runAll();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
    
    printf("Test Summaries:\r\n");
    test.printSummary();

    return 0;
}
