#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "mbed.h"
#include "stdio.h"
#include "uTensor_util.hpp"
#include "tensor.hpp"
#include "context.hpp"
#include "model.hpp"


Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char** argv) {
  ON_ERR(bd.init(), "SDBlockDevice init ");
  ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");
  init_env();

  Context ctx;
  Tensor* input = new RamTensor<uint8_t>({784});
  get_quantized_graph_ctx(ctx, input);
  ctx.eval();


  printf("\r\ndone...\r\n");

  ON_ERR(fs.unmount(), "fs unmount ");
  ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
}
