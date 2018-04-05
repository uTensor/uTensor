#include "FATFileSystem.h"
#include "uTensor/util/uTensor_util.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/context.hpp"
// #include "mlp_test.hpp"
#include "examples/deep_mnist_mlp.hpp"
#include "SDBlockDevice.h"
#include "mbed.h"
#include <stdio.h>

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char** argv) {
  ON_ERR(bd.init(), "SDBlockDevice init ");
  ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");
  init_env();

  printf("Deep MLP on Mbed (Trained with Tensorflow)\r\n\r\n");
  printf("running deep-mlp...\r\n");

  int prediction = runMLP("/fs/testData/deep_mlp/import-Placeholder_0.idx");
  printf("prediction: %d\r\n\r\n\r\n\r\n", prediction);


  printf("\r\ndone...\r\n");

  ON_ERR(fs.unmount(), "fs unmount ");
  ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
}
