# uTensor - Test Release
[![CircleCI](https://circleci.com/gh/uTensor/uTensor.svg?style=svg)](https://circleci.com/gh/uTensor/uTensor)
Note: If you are looking for stable releases, checkout master.

## Release Note
- Updated uTensor ReadMe
- Updated uTensor-CLI ReadMe
- ROM Tensor support


## Introduction
### What is it?
uTensor is an extremely light-weight machine learning inference framework built on Mbed and Tensorflow. It consists of a runtime library and an offline tool. The total size of graph definition and algorithm implementation of a 3-layer MLP produced by uTensor is less than 32kB in the resulting binary (excluding the weights).

### How does it work?
<div><img src=docs/img/uTensorFlow.jpg width=600 align=center/></div>

A model is constructed and trained in Tensorflow. uTensor takes the model and produces a .cpp and .hpp file. These files contains the generated C++11 code needed for inferencing. Working with uTensor on the embedded side is as easy as copy-and-paste. The function interface looks like this:

```
#include "models/deep_mlp.hpp"
...
Context ctx;  //creating a context
...
//preparing for the input tensor
...
get_deep_mlp_ctx(Context& ctx, Tensor* input_0);  //perform inference
ctx.eval();
S_TENSOR prediction = ctx.get({"y_pred:0"});  //getting the result
```
The .hpp and .cpp files can be generated given a model (protocal buffer) file, for example:

```
$ utensor-cli deep_mlp.pb --output-nodes=y_pred
...
... Generate weight file: models/deep_mlp_weight.hpp
... Generate header file: models/deep_mlp.hpp
... Generate source file: models/deep_mlp.cpp
```

### What's supported?
The project is work-in-progress. Here are the operators, of their __quantized__ versions, that are currently avaliable:

- Add
- ArgMax
- Dropout
- MatMal
- Max
- Min
- Placeholder
- Quantization Ops
- ReLu
- Reshape

## Quick Start
### Hardware

uTensor should support any [Mbed enabled board](https://os.mbed.com/platforms/?mbed-os=21&mbed-os=22&mbed-os=25&mbed-os=26&mbed-os=33) that has sufficient memory (128+ kB RAM and 512kB+ flash recommended). However, these two boards are popular among the core developers:

- [FRDM-K66F](https://os.mbed.com/platforms/FRDM-K66F/): reference development environment
- [DISCO-F413ZH](https://os.mbed.com/platforms/ST-Discovery-F413H/): a good demo/application prototyping platform, wi-fi
- Any [Mbed board](https://os.mbed.com/platforms/?mbed-os=21&mbed-os=22&mbed-os=25&mbed-os=26&mbed-os=33&mbed-os=34) with sufficient memory

### The Environment

  - [Mbed-CLI](https://github.com/ARMmbed/mbed-cli)
  - [uTensor-CLI](https://github.com/uTensor/utensor_cgen)


### Getting Started
#### [Creating a New Project](https://blog.hackster.io/simple-neural-network-on-mcus-a7cbd3dc108c)
An end-to-end tutorial going from training a neural network to deployment on a device. You will need a [K66F](https://os.mbed.com/platforms/FRDM-K66F/) or a smiliar board for this tutorial.

#### [MNIST Touch Screen](https://github.com/uTensor/utensor-mnist-demo)
The example uses a 3-layer MLP trained on the MNIST dataset. The touch screen input is fed into the neural network for processing and the result is printed on the screen.

#### [The Activity of Daily Living (ADL)](https://github.com/uTensor/ADL_demo)
This example shows how to buffer time-series data into batches of snapshots. These snapshots are then fed into the neural network for inferencing. The model a small multi-layer MLP trained on the [ADL dataset](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer).


## Development
uTensor is young and under going rapid development. Many exciting features are on the way:

- Convolution
- Pooling
- CMSIS-NN integration
- Smaller binary
- More efficient Run-time

You can also check the [project page](https://github.com/orgs/uTensor/projects) for the latest progress. If you'd like to take part in this project, please have a look at our **[contributor guide](contribution_guide.md)** and feel free to reach out to us.

## Further Reading
- [Why Edge Computing](https://towardsdatascience.com/why-machine-learning-on-the-edge-92fac32105e6)
- [Why the Future of Machine Learning is Tiny](https://petewarden.com/2018/06/11/why-the-future-of-machine-learning-is-tiny/)
- [TensorFlow](https://www.tensorflow.org)
- [Mbed](https://developer.mbed.org)
- [Node-Viewer](https://github.com/neil-tan/tf-node-viewer/)
- [How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
- [mxnet Handwritten Digit Recognition](https://mxnet.incubator.apache.org/tutorials/python/mnist.html)
