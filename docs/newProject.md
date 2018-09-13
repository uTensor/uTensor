# Creating a New Project
This is a quick start guide highlighting the key steps to create your own uTensor Project from scretch. We will use the MNIST MLP training script to generate the model file. The flow is based on Mbed-CLI and an UNIX-like environment. The target is [FRDM-K66F](https://os.mbed.com/platforms/FRDM-K66F/).

## Requirement
- [Mbed Cli](https://github.com/ARMmbed/mbed-cli)
- [uTensor Cli](https://github.com/uTensor/utensor_cgen)
- Tensorflow (comes with uTensor Cli installation)
- [FRDM-K66F](https://os.mbed.com/platforms/FRDM-K66F/), or any Mbed board with sufficient memory
- A micro-SD card (32GB or less, for demostration purpose)

## Setting it up
Let's start with creating a **new Mbed project**.

```
$ mbed new my_uTensor
$ cd my_uTensor/
$ ls
mbed-os          mbed-os.lib      mbed_settings.py
```
We will need the **uTensor runtime library**. It contains all the function implementations that will be linked during the compilation time.

```
$ mbed add https://github.com/uTensor/uTensor
$ ls
mbed-os          mbed_settings.py uTensor.lib
mbed-os.lib      uTensor
```
For development purpose, it is useful to have SD accecss:

```
mbed add https://github.com/ARMmbed/sd-driver/#c46d0779e7354169e5a5ad0a84dc0346f55ab3e1
```
## Generating the Model File
You may have your own step to obtain the .pb file. For the sake of illustraion, let's use the MNIST **training** script:

```
$ wget https://raw.githubusercontent.com/uTensor/utensor-mnist-demo/master/tensorflow-models/deep_mlp.py
$ ls
deep_mlp.py      mbed-os.lib      uTensor
mbed-os          mbed_settings.py uTensor.lib
```

Genrate **.pb** file by running the training script.

```
$ python deep_mlp.py
...
step 19000, training accuracy 0.92
step 20000, training accuracy 0.94
test accuracy 0.9274
saving checkpoint: chkps/mnist_model
Converted 6 variables to const ops.
written graph to: mnist_model/deep_mlp.pb
the output nodes: ['y_pred']
```
In this example, the .pb file is located under the mnist_model folder.

```
$ ls mnist_model/
deep_mlp.pb
```

## Code Generation
The uTensor-Cli can be installed with:

```pip install utensor-cli```

Fire up your **utensor-cli**. Depends on your installation, you may need to activate your **virtual environment**. Supply your .pb file and the name of your output node.

```
$ utensor-cli convert mnist_model/deep_mlp.pb --output-nodes=y_pred
... Applying sort_by_execution_order
... Transforming graph: mnist_model/deep_mlp.pb
... Applying quantize_weights
... Applying quantize_nodes
... Applying sort_by_execution_order
... Graph transormation done
... Generate weight file: models/deep_mlp_weight.hpp
... Generate header file: models/deep_mlp.hpp
... Generate source file: models/deep_mlp.cpp
```
The graph definition and model parameters have now been compiled into C++ source and header files under `models/`, readily to be imported into your projects.

## Adding the main.cpp
Create a main.cpp file at the project root, and fill it with the following code:

```
#include "models/deep_mlp.hpp"
#include "SDBlockDevice.h"
#include "tensorIdxImporter.hpp"
#include "tensor.hpp"
#include "FATFileSystem.h"
#include "mbed.h"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

void run_mlp(){
  TensorIdxImporter t_import;
  Tensor* input_x = t_import.float_import("/fs/tmp.idx");
  Context ctx;

  get_deep_mlp_ctx(ctx, input_x);
  S_TENSOR pred_tensor = ctx.get("y_pred:0");
  ctx.eval();

  int pred_label = *(pred_tensor->read<int>(0, 0));
  printf("Predicted label: %d\r\n", pred_label);

}

int main(void) {
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");
  
  ON_ERR(bd.init(), "SDBlockDevice init ");
  ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

  run_mlp();
  
  ON_ERR(fs.unmount(), "fs unmount ");
  ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

  return 0;
}
```

## Prepare the SD card
In this example, we will place the input data on the SD card. It is an .idx file containing the first sample of the MNIST dataset, representing number 7.

You will need a 32GB or less FAT32 formated SD card.

- Copy this [file](https://github.com/uTensor/utensor-helloworld/blob/master/sdcard/tmp.idx) onto the root of your SD card. It is going to be our test input.
- Insert the SD card into the board.

## Compile
Use the mbed configuration supplied by uTensor.

```
cp uTensor/mbed_app.json .
```

Now, you are ready to compile the project. The command consists of 4 parts:

- -m: name of the target; auto means use whatever target is connected via USB.
- -t: GCC_ARM is selected at our compiler
- --profile: supplied by uTensor. This include settings to enable C++11
- -f: flash onto the board when done. Forget about this if you are on cloud9, download and drag-and-drop instead.


```
$ mbed compile -m auto -t GCC_ARM --profile=uTensor/build_profile/release.json -f
...
Image: ./BUILD/DISCO_F413ZH/GCC_ARM/my_uTensor.bin
[mbed] Detected "DISCO_F413ZH" connected to "/Volumes/DIS_F413ZH" and using com port "/dev/tty.usbmodem1413"
```

## Expected Output
Connect to the board via CoolTerm using 115200 as the baudrate. Here is what you should see:

```
Simple MNIST end-to-end uTensor cli example (device)
Predicted label: 7
```

**Congradulations!** You have deployed a neural network on a MCU from scretch!