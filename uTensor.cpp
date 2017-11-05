#include "uTensor.hpp"


uTensor::uTensor():
    pc(USBTX, USBRX, 9600),
    bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS),
    fs("fs"){
        ON_ERR(bd.init(), "SDBlockDevice init ");
        ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

        printf("Deep MLP on Mbed (Trained with Tensorflow)\r\n\r\n");
        printf("running deep-mlp...\r\n");

        int prediction = runMLP("/fs/testData/deep_mlp/import-Placeholder_0.idx");
        printf("prediction: %d\r\n", prediction);

        //In [24]: tf.get_default_graph().get_tensor_by_name("import/y_pred:0").eval(feed_dict={x: mnist.test.images[0:1]})
        //Out[24]: array([7])

    }

uTensor::~uTensor(){

//    ON_ERR(fs.unmount(), "fs unmount ");
//    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
}
