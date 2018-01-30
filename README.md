
# uTensor

## Introduction

  uTensor is an extremely light-weight Deep-Learning Inference framework built on Mbed and Tensorflow:
  
  - TensorFlow to uTensor exporter (planned)
  - Tensor Classes
  - Operators Classes
  - Context, a resource and graph management class

  This project is under going constant development. APIs are expected to update rapidly.

## Requirement

- [Mbed CLI](https://github.com/ARMmbed/mbed-cli)
- [Tensorflow](https://www.tensorflow.org/install/)
- [tf-node-viewer](https://github.com/neil-tan/tf-node-viewer) (Optional, for graph-weight extraction)
- Mbed-os 5.6+ compatible [boards](https://os.mbed.com/platforms/?mbed-os=25) with at least 256kb of RAM
- SD Card (Must be LESS than 32 GB)
- SD Card reader for the board (Optional if built into the board)

## Finding your target name

`mbed detect` to see which target is connect to the board

`mbedls -l` to list all supported targets

## Configure

See mbed_app.json

## Build Steps

1. Clone the repository
2. Run `mbed deploy` to download all referenced libraries
3. Insert the prepared SD card to the board (see SD Card Preparation Section)
4. Use `mbed compile -t GCC_ARM -m NUCLEO_F767ZI --profile=./build_profile/release.json` to build for ST NUCLEO F767ZI. Or, `mbed compile -t GCC_ARM -m NUCLEO_F767ZI --profile=./build_profile/release.json -f` to compile and flash

## SD Card Preparation
The test data has to be loaded to the SD card for the default binary to run:

1. Install python dependencies `pip install -r requirements.txt` (Note: may have to use `pip3`)
1. Go to the `[project]\TESTS\scripts` folder
1. Run `python3 compileTestData.py`. This will create `[project]\TESTS\scripts\testData` directory.
1. Copy `[project]\TESTS\scripts\testData` to the root of your SD card.

## Expected Output
The quantized weight and input data are stored in the SD. Setting the serial baud rate to 115200, here is what you should see:

```
Deep MLP on Mbed (Trained with Tensorflow)

running deep-mlp...
PASSED 0.00000000

prediction: 7
```
Currently, the binary runs the first sample of the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) which contains a handwritten digit of number 7. Ths network architecture is a 3-layer Relu based MLP, as shown below:

![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png "mxnet Handwritten Digit Recognition")

 The related Tensorflow training script please refer to the [node-viewer](https://github.com/neil-tan/tf-node-viewer/blob/master/deep_mlp.py) project.
 
## Exporting to uTensor
  
Please refer to [uTensor-CLI](https://github.com/utensor/utensor_cgen)
  
   
## Reference

- [TensorFlow](https://www.tensorflow.org)
- [Mbed](https://developer.mbed.org)
- [Node-Viewer](https://github.com/neil-tan/tf-node-viewer/)
- [How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
- [mxnet Handwritten Digit Recognition](https://mxnet.incubator.apache.org/tutorials/python/mnist.html)


