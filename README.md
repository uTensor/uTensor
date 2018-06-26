# uTensor - Test Release
## Introduction
### What is it?
uTensor is an extremely light-weight machine learning inference framework built on Mbed and Tensorflow. The project contains a runtime library and an offline tool. The total size of graph definition and algorithm implementation of a 3-layer MLP produced by uTensor is less than 32kB in the resulting binary (excluding the weights).

### How does it work?
<div><img src=docs/img/uTensorFlow.jpg width=600 align=center/></div>

A model is constructed and trained in Tensorflow. uTensor takes the model and produces a .cpp and .hpp file. These files contains the generated C++11 code needed for inferencing. Working with uTensor on the embedded side is as easy as copy-and-past. The function interface looks like this:

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

- [DISCO-F413ZH](https://os.mbed.com/platforms/ST-Discovery-F413H/): a good demo/application prototyping platform, wi-fi
- [K64F](https://os.mbed.com/platforms/FRDM-K64F/): rock-solid development environment

You will need a FAT32 formated SD card. Please note, the size of the SD card has to be less than 32GB. An SD card will be made optional in the future releases.

### The Environment
There are two flows to get started with uTensor. For Windows users, please choose the Cloud9 flow as shown below.

- The [Cloud9 Flow](https://github.com/uTensor/cloud9-installer)
  - Requires Amazon Cloud9 Account
  - Does not support runtime debugging
- Local Installation
  - Requires [Mbed-CLI](https://github.com/ARMmbed/mbed-cli) installation (Python 2)
  - Requires [uTensor-CLI](https://github.com/uTensor/utensor_cgen) installation

### The Examples
#### [MNIST Touch Screen](https://github.com/uTensor/utensor-mnist-demo)
The example uses a 3-layer MLP trained on the MNIST dataset. The touch screen input is fed into the neural network for processing and the result is printed on the screen.

#### [The Activity of Daily Living (ADL)](https://github.com/uTensor/ADL_demo)
This example shows how to buffer time-series data into batches of snapshots. These snapshots are then fed into the neural network for inferencing. The model a small multi-layer MLP trained on the [ADL dataset](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer).

### Creating a New Project

  Please refer to this [guide](docs/newProject.md) for instructions on creating your own project from stretch.

## Development
uTensor is young and under going rapid development. Many exciting features are on the way:

- Convolution
- Pooling
- SD cards optional: ability to store weights in on-board flash
- CMSIS-NN integration
- Smaller binary

You can also check the [project page](https://github.com/orgs/uTensor/projects) for the latest process. If you'd like to take part in this project, please have a look at our contributor guide and feel free to reach out to us.

## Release Note
- Updated uTensor ReadMe
- Updated uTensor-CLI ReadMe
- Added Contributor Guide
- Dropout Support
