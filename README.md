# uTensor (Deep MLP) Hello World

## Introduction

  uTensor is an extremely light-weight Deep-Learning Inference framework built on Mbed and Tensorflow. The project contains a Mbed importable runtime library and [utensor-cli](https://github.com/uTensor/utensor_cgen), an offline-tool which generates embedded C++ code base on supplied quantized-inference-graph.
  
  The project current consists of:
 
### Offline Tool
  
  - TensorFlow to uTensor exporter ([utensor-cli](https://github.com/uTensor/utensor_cgen))

### Mbed Runtime Library

  - Tensor Classes
	- Data Holder
	- Virtual Memory

  - Operators Classes
	- C reference implementation
	- Basic operators: MatMal, Add, ReLu, Reshape, Max, Min, ArgMax, Quantization Ops, etc

  - Context Class
	- A resource management class
	- An interface to utensor-cli's code generation
	- Describes a graph

  This project is under going constant development. APIs are expected to update rapidly.
<<<<<<< HEAD
  
## Overview
  This document contains the steps you would need to build an uTensor application from ground-up. The application implements a simple 3-layer MLP trained for the MNIST dataset, a hand-written digit recognizer:

  ![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png "mxnet Handwritten Digit Recognition")
  
Topics to be covered:
  
  - Setting up a Mbed project with uTensor
  - Convert a quantized graph to C++ code
  - Add main.cpp to perform inference
  - Running on device

    The example repository created using steps presented below can be found [HERE](https://github.com/neil-tan/utensor-helloworld).
    
  We are working to release an Mbed-online-Simulator version of the example. Stay-tuned!

## Requirement

- [Mbed CLI](https://github.com/ARMmbed/mbed-cli)
- Mbed-os 5.6+ compatible [boards](https://os.mbed.com/platforms/?mbed-os=25) with at least 256kb of RAM
- Python3
- CoolTerm
- FAT32 formated SD card (must be less than 32 GB)
- SD Card reader for the board (Optional if built into the board)


## Project Setup
  The default targets for uTensor are Mbed enabled boards. Mbed-cli is the build we use it to:

  - Manage Mbed projects
  - Import libraries
  - Compile the code
  - Flash onto devices

  Let's start by creating a project folder and initial an Mbed project:
  
  ```
  $ mkdir	helloworld
  $ cd helloworld
  $ mbed new .
  [mbed] Creating new program "helloworld" (git)
[mbed] Adding library "mbed-os" from "ssh://git@github.com/ARMmbed/mbed-os.git" at branch latest
[mbed] Updating reference "mbed-os" -> "https://github.com/ARMmbed/mbed-os/..."
  $ ls #See what `mbed new .` does
  mbed-os          mbed-os.lib      mbed_settings.py
  ```
  
  Next, we would like to add the uTensor runtime-library to the project:

  ```
  $ mbed add https://github.com/uTensor/uTensor
  [mbed] Updating library "uTensor" to latest revision in the current branch
[mbed] Updating reference "uTensor" -> "https://github.com/uTensor/uTensor/...
  ```
  
  Because uTensor usually require a file system to access the model parameters, we would have to add a filesystem driver. In this case, SD driver is used:

  ```
  $ mbed add https://github.com/ARMmbed/sd-driver/#c46d0779e7354169e5a5ad0a84dc0346f55ab3e1
  [mbed] Updating library "sd-driver" to rev #c46d0779e735
  ```
  
  Now, all required libraries have been added. As a good measure, we are issuing `mbed deploy` to ensure all the references are in good shape.
  
	```
  $ mbed deploy
  [mbed] Updating library "mbed-os" to rev #96d9a00d0a1d
  [mbed] Updating library "sd-driver" to rev #c46d0779e735
  [mbed] Updating library "uTensor" to rev #1bdf2b3d5628
	```
	
Finally, we would like to use uTensor's application profile as a starting template for our configurations:

```
$ cp uTensor/mbed_app.json ./
```
	
## Graph to C++
  This section shows how one would use utensor-cli to generate the C++ implementation of the model given a quantized graph trained within Tensorflow. Here, we would a graph we prepared for illustration purpose. Overview of the steps are:
  
 - Install utensor-cli
 - Running utensor-cli to convert the graph to C++
 - Copying files
	 - Copy model parameters to the device filesystem
	 - Copy the C++ model files to your project

### Installing utensor-cli
  It's recommended that we do this in a virtual environment:

```
 # Do this at the same level as your project folder
 $ cd ..
 $ ls
helloworld
 $ mkdiri py3_venv
 $ python3 -m venv ./py3_venv/ut
 $ source ./py3_venv/ut/bin/activate
 (ut) $ ls
helloworld  py3_venv
```

 Clone and install utensor-cli:

 ```
  (ut) $ git clone https://github.com/uTensor/utensor_cgen
  (ut) $ cd utensor_cgen
  (ut) $ pip install utensor_cgen
  Collecting utensor_cgen
  Downloading utensor_cgen-0.1.2.tar.gz
  ...
  Successfully installed ...
 ```
### Convert the graph to C++
 Generating C++ files from a pre-trained and pre-quantized graph:

 ```
  (ut) $ utensor-cli tests/deep_mlp/quant_mnist.pb
  	...
	saving constants/quant_mnist/OuputLayer_Variable_1_0.idx
	saving constants/quant_mnist/y_pred_dimension_0.idx
	Generate header file: models/quant_mnist.hpp
	Generate source file: models/quant_mnist.cpp
 ```
   This step saves model parameters in `./constants/`. The `models/quant_mnist.hpp` and `models/quant_mnist.cpp` are to be imported to our Mbed project.
 
### Copying files

  Model parameters, in form of idx files, needs to be loaded by your device during runtime. In this example, we store them on a SD card. The generated C++ graph implementation needs to be placed in your Mbed project folder during compilation-time:

- Copy ./constants to the root of your FAT32 formated SD card

-  Copy ./model to your Mbed project root

```
#You may deactiavte your virtual environment now
(ut) $ deactivate
$ cp -r models ../helloworld/
$ cd ../helloworld/
$ ls
mbed-os          mbed-os.lib      mbed_settings.py models           sd-driver        sd-driver.lib    uTensor          uTensor.lib
```
## Add main.cpp to perform inference
Create a main.cpp in your Mbed project and add the following code to it:

```
#include "quant_mnist.hpp"
#include "tensorIdxImporter.hpp"
#include "tensor.hpp"
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "mbed.h"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

void run_mlp(){
  TensorIdxImporter t_import;
  Tensor* input_x = t_import.float_import("/fs/tmp.idx");
  Context ctx;

  get_quant_mnist_ctx(ctx, input_x);
  S_TENSOR pred_tensor = ctx.get("y_pred:0");
  ctx.eval();

  int pred_label = *(pred_tensor->read<int>(0, 0));
  printf("Predicted label: %d\r\n", pred_label);

}

int main(void) {
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");
  
  ON_ERR(bd.init(), "SDBlockDevice init ");
  ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

  init_env();
  run_mlp();
  
=======
  
## Overview
  This document contains the steps you would need to build an uTensor application from ground-up. The application implements a simple 3-layer MLP trained for the MNIST dataset, a hand-written digit recognizer:

  ![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png "mxnet Handwritten Digit Recognition")
  
Topics to be covered:
  
  - Setting up a Mbed project with uTensor
  - Convert a quantized graph to C++ code
  - Add main.cpp to perform inference
  - Running on device

    The example repository created using steps presented below can be found [HERE](https://github.com/neil-tan/utensor-helloworld).
    
  We are working to release an Mbed-online-Simulator version of the example. Stay-tuned!

## Requirement

- [Mbed CLI](https://github.com/ARMmbed/mbed-cli)
- Mbed-os 5.6+ compatible [boards](https://os.mbed.com/platforms/?mbed-os=25) with at least 256kb of RAM
- Python3
- CoolTerm
- FAT32 formated SD card (must be less than 32 GB)
- SD Card reader for the board (Optional if built into the board)


## Project Setup
  The default targets for uTensor are Mbed enabled boards. Mbed-cli is the build we use it to:

  - Manage Mbed projects
  - Import libraries
  - Compile the code
  - Flash onto devices

  Let's start by creating a project folder and initial an Mbed project:
  
  ```
  $ mkdir	helloworld
  $ cd helloworld
  $ mbed new .
  [mbed] Creating new program "helloworld" (git)
[mbed] Adding library "mbed-os" from "ssh://git@github.com/ARMmbed/mbed-os.git" at branch latest
[mbed] Updating reference "mbed-os" -> "https://github.com/ARMmbed/mbed-os/..."
  $ ls #See what `mbed new .` does
  mbed-os          mbed-os.lib      mbed_settings.py
  ```
  
  Next, we would like to add the uTensor runtime-library to the project:

  ```
  $ mbed add https://github.com/uTensor/uTensor
  [mbed] Updating library "uTensor" to latest revision in the current branch
[mbed] Updating reference "uTensor" -> "https://github.com/uTensor/uTensor/...
  ```
  
  Because uTensor usually require a file system to access the model parameters, we would have to add a filesystem driver. In this case, SD driver is used:

  ```
  $ mbed add https://github.com/ARMmbed/sd-driver/#c46d0779e7354169e5a5ad0a84dc0346f55ab3e1
  [mbed] Updating library "sd-driver" to rev #c46d0779e735
  ```
  
  Now, all required libraries have been added. As a good measure, we are issuing `mbed deploy` to ensure all the references are in good shape.

	```
		
	$ mbed deploy
	[mbed] Updating library "mbed-os" to rev #96d9a00d0a1d
	[mbed] Updating library "sd-driver" to rev #c46d0779e735
	[mbed] Updating library "uTensor" to rev #1bdf2b3d5628
	```
	
  Finally, we would like to use uTensor's application profile as a starting template for our configurations:

	```
	$ cp uTensor/mbed_app.json ./
	```

## Graph to C++
  This section shows how one would use utensor-cli to generate the C++ implementation of the model given a quantized graph trained within Tensorflow. Here, we would a graph we prepared for illustration purpose. Overview of the steps are:
  
 - Install utensor-cli
 - Running utensor-cli to convert the graph to C++
 - Copying files
	 - Copy model parameters to the device filesystem
	 - Copy the C++ model files to your project

### Installing utensor-cli
  It's recommended that we do this in a virtual environment:

```
 # Do this at the same level as your project folder
 $ cd ..
 $ ls
helloworld
 $ mkdiri py3_venv
 $ python3 -m venv ./py3_venv/ut
 $ source ./py3_venv/ut/bin/activate
 (ut) $ ls
helloworld  py3_venv
```

 Clone and install utensor-cli:

 ```
  (ut) $ git clone https://github.com/uTensor/utensor_cgen
  (ut) $ cd utensor_cgen
  (ut) $ pip install utensor_cgen
  Collecting utensor_cgen
  Downloading utensor_cgen-0.1.2.tar.gz
  ...
  Successfully installed ...
 ```
### Convert the graph to C++
 Generating C++ files from a pre-trained and pre-quantized graph:

 ```
  (ut) $ utensor-cli tests/deep_mlp/quant_mnist.pb
  	...
	saving constants/quant_mnist/OuputLayer_Variable_1_0.idx
	saving constants/quant_mnist/y_pred_dimension_0.idx
	Generate header file: models/quant_mnist.hpp
	Generate source file: models/quant_mnist.cpp
 ```
   This step saves model parameters in `./constants/`. The `models/quant_mnist.hpp` and `models/quant_mnist.cpp` are to be imported to our Mbed project.
 
### Copying files

  Model parameters, in form of idx files, needs to be loaded by your device during runtime. In this example, we store them on a SD card. The generated C++ graph implementation needs to be placed in your Mbed project folder during compilation-time:

- Copy ./constants to the root of your FAT32 formated SD card

-  Copy ./model to your Mbed project root

```
#You may deactiavte your virtual environment now
(ut) $ deactivate
$ cp -r models ../helloworld/
$ cd ../helloworld/
$ ls
mbed-os          mbed-os.lib      mbed_settings.py models           sd-driver        sd-driver.lib    uTensor          uTensor.lib
```
## Add main.cpp to perform inference
Create a main.cpp in your Mbed project and add the following code to it:

```
#include "quant_mnist.hpp"
#include "tensorIdxImporter.hpp"
#include "tensor.hpp"
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "mbed.h"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

void run_mlp(){
  TensorIdxImporter t_import;
  Tensor* input_x = t_import.float_import("/fs/tmp.idx");
  Context ctx;

  get_quant_mnist_ctx(ctx, input_x);
  S_TENSOR pred_tensor = ctx.get("y_pred:0");
  ctx.eval();

  int pred_label = *(pred_tensor->read<int>(0, 0));
  printf("Predicted label: %d\r\n", pred_label);

}

int main(void) {
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");
  
  ON_ERR(bd.init(), "SDBlockDevice init ");
  ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

  init_env();
  run_mlp();
  
  ON_ERR(fs.unmount(), "fs unmount ");
  ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

  return 0;
}

```
Connect your device to your machine and compile your project with C++ profile:

```
#let mbed-cli set compilation target base on the device you have connected to you machine
$ mbed target auto
[mbed] auto now set as default target in program "helloworld"
$ mbed compile -t GCC_ARM --profile=./uTensor/build_profile/release.json
Compile [100.0%]: quantization_utils.cpp
...
Link: helloworld
Elf2Bin: helloworld
+----------------------+--------+-------+-------+
| Module               |  .text | .data |  .bss |
+----------------------+--------+-------+-------+
| [fill]               |    394 |     7 |  2219 |
| [lib]/c.a            |  68359 |  2548 |   127 |
| [lib]/gcc.a          |   7200 |     0 |     0 |
| [lib]/misc           |    248 |     8 |    28 |
| [lib]/nosys.a        |     32 |     0 |     0 |
| [lib]/stdc++.a       | 171325 |   141 |  5676 |
| main.o               |    101 |     0 |     1 |
| mbed-os/drivers      |     52 |     0 |     0 |
| mbed-os/features     |    107 |     0 |   188 |
| mbed-os/hal          |   1361 |     4 |    66 |
| mbed-os/platform     |   1624 |     4 |   314 |
| mbed-os/rtos         |   8712 |   168 |  5989 |
| mbed-os/targets      |   4998 |    12 |   384 |
| models/quant_mnist.o |     36 |     0 |     1 |
| uTensor/examples     |     36 |     0 |     1 |
| uTensor/uTensor      |    413 |     0 |     6 |
| Subtotals            | 264998 |  2892 | 15000 |
+----------------------+--------+-------+-------+
Total Static RAM memory (data + bss): 17892 bytes
Total Flash memory (text + data): 267890 bytes

Image: ./BUILD/K64F/GCC_ARM/helloworld.bin
```

## Running on device
- Download [the handwritten sample](https://github.com/uTensor/uTensor/blob/master/TESTS/scripts/PRE-GEN/deep_mlp/import-Placeholder_0.idx) to the root of your SD card and rename it as `tmp.idx`.

- Insert the prepared SD card into your device.

- Connect your device to your machine and invoke `mbed compile` again with the `-f` argument:

```
mbed compile -t GCC_ARM --profile=./uTensor/build_profile/release.json -f
```

- Launch CoolTerm and connect to your device with baudrate of 115200. Here is the expected output:

```
Simple MNIST end-to-end uTensor cli example (device)
Predicted label: 7
```

## Further Reading

- [TensorFlow](https://www.tensorflow.org)
- [Mbed](https://developer.mbed.org)
- [Node-Viewer](https://github.com/neil-tan/tf-node-viewer/)
- [How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
- [mxnet Handwritten Digit Recognition](https://mxnet.incubator.apache.org/tutorials/python/mnist.html)
