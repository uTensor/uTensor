#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "mbed.h"
#include "stdio.h"
#include "uTensor_util.hpp"
#include "tensor.hpp"
#include "tensorIdxImporterTests.hpp"
#include "context.hpp"
#include "MatrixTests.hpp"
#include "ArrayTests.hpp"
#include "MathTests.hpp"
#include "NnTests.hpp"
#include "tensor_test.hpp"
#include "context_test.hpp"
#include "mlp_test.hpp"
#include "deep_mnist_mlp.hpp"
//#include "deep_mnist_mlp.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char** argv) {
  ON_ERR(bd.init(), "SDBlockDevice init ");
  ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

  printf("Deep MLP on Mbed (Trained with Tensorflow)\r\n\r\n");
  printf("running deep-mlp...\r\n");

  int prediction = runMLP("/fs/testData/deep_mlp/import-Placeholder_0.idx");
  printf("prediction: %d\r\n", prediction);
   idxImporterTest idxTest;
   idxTest.runAll();


   printf("running matrix test ...\r\n");
   matrixOpsTest matrixTests;
   matrixTests.runAll();
   printf("IDX import:\r\n");
   idxTest.printSummary();
   printf("Matrix: \r\n");
   matrixTests.printSummary();

   ArrayOpsTest arrayTests;
   arrayTests.runAll();
   printf("Array: \r\n");
   arrayTests.printSummary();

   printf("Math: \r\n");
   MathOpsTest mathTests;
   mathTests.runAll();
   printf("Math result...\r\n");
   mathTests.printSummary();

   printf("NnOpS: \r\n");
   NnOpsTest nnTest;
   nnTest.runAll();
   printf("Nn Ops result...\r\n");
   nnTest.printSummary();

   printf("Transformation test: \r\n");
   transTest tTest;
   tTest.runAll();
   printf("Transformation result...\r\n");
   tTest.printSummary();

   printf("Context test: \r\n");
   contextTest ctxTest;
   ctxTest.runAll();
   printf("Context result...\r\n");
   ctxTest.printSummary();


  //  printf("mlp test: \r\n");
  //  mlpTest mlpt;
  //  mlpt.runAll();
  //  printf("mlp result...\r\n");
  //  mlpt.printSummary();


   printf("tesnor test: \r\n");
   tensorTest tenT;
   tenT.runAll();
   printf("tensor result: \r\n");
   tenT.printSummary();

    printf("Test framework validation:\r\n");
   if(arrayTests.getTestResult("quantize_v2") == Test_Result::PASSED) { printf("Passed : %.4f\r\n", arrayTests.getTestTime("quantize_v2"));}
   if(arrayTests.getTestResult("dequantize") == Test_Result::PASSED) {printf("Passed : %.4f\r\n", arrayTests.getTestTime("dequantize"));}
   if(arrayTests.getTestResult("reshape") == Test_Result::PASSED) {printf("Passed : %.4f\r\n", arrayTests.getTestTime("reshape"));}


  //In [24]: tf.get_default_graph().get_tensor_by_name("import/y_pred:0").eval(feed_dict={x: mnist.test.images[0:1]})
  //Out[24]: array([7])

  printf("\r\ndone...\r\n");

  ON_ERR(fs.unmount(), "fs unmount ");
  ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
}
