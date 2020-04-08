# uTensor - Test Release
[![CircleCI](https://circleci.com/gh/uTensor/uTensor.svg?style=svg)](https://circleci.com/gh/uTensor/uTensor)
Note: If you are looking for stable releases, checkout master.

## Introduction

### What is it?
uTensor is an extremely light-weight machine learning inference framework built on Tensorflow and optimized for Arm targets. It consists of a runtime library and an offline tool that handles most of the model translation work. This repo holds the core runtime and some example implementations of operators, memory managers/schedulers, and more, and the size of the core runtime is only ~2KB!

| Module                       |         .text |       .data |        .bss |
|------------------------------|---------------|-------------|-------------|
| uTensor/src/uTensor/core     |   1275(+1275) |       4(+4) |     28(+28) |
| uTensor/src/uTensor/tensors  |     791(+791) |       0(+0) |       0(+0) |


### How does the uTensor workflow work?
<div><img src=docs/img/uTensorFlow.jpg width=600 align=center/></div>

A model is constructed and trained in Tensorflow. uTensor takes the model and produces a .cpp and .hpp file. These files contains the generated C++11 code needed for inferencing. Working with uTensor on the embedded side is as easy as copy-and-paste.

### How does the uTensor runtime work?
TODO


## Release Note
The rearchitecture is fundamentally centered around a few key ideas, and the structure of the code base and build tools naturally followed.
Old key points:
- Tensors describe how data is accessed and where from
  - Performance of ops depends on which tensors are used
- Operators are Tensor agnostic
  - High performance ops can fetch blocks of data at once
- Strive for low total power in execution
- Low static and dynamic footprint, be small
  - Low cost per Tensor throughout the entire system, since most generated models have 100+ including intermediates, also impacts dynamic footprint
  - Lightweight class hierarchy
  - Duh

New additional key ideas:
- System safety
  - All tensor metadata and actual data are owned in dedicated regions
    - This can either be user provided, or one we create
  - We can guarantee that runtime will use no more than N bytes of RAM at code gen time or at compile time!
  - Generally should not collide with userspace or system space memory, i.e. dont share heaps
  - Generally implications: a safe runtime means we can safely update models remotely
  - As many compile time errors as possible!
    - Mismatched inputs, outputs, or numbers
    - wrong sizes used
    - Impossible memory accesses
    - etc.
- Clear, Concise, and Debuggable
  - Previous iteration of uTensor relied almost too heavily on codegen, making changes to a model for any reason was near impossible
  - A developer should be able to make changes to the model without relying on code gen
  - A developer should be able to look at a model file and immediately understand what the graph looks like, without a massive amound of jumping around
  - Default tensor interface should behave like a higher level language, but exploit the speed of C++
    - Generally: No more pointer bullshit! C is super error prone, fight me
      - Only specialized operators have access to raw data blocks, and these ops will be wicked fast
  - Extensible, configurable, and optimize-outable error handling
  - GDB debugging IS NOW TRIVIAL

As mentioned before, these key ideas need to be reflected not only in the code, but in the code structure in such a way that it is Maintainable, Hackable, and User-extensible. Pretty much everything in the uTensor runtime can be divided into two components: core, and everything else. The core library contains all the deep low level functionality needed for the runtime to make the above guarantees, as well as the interfaces required for concrete implementation. Furthermore, the overhead of this core engine should be negligible relative to the system operation. Everything not in the core library really should just be thought of a reasonable defaults. For example, tensor implementations, default operators, example memory allocators, or even possible logging systems and error handlers. These modules should be the primary area for future optimization, especially before model deployment.

## High level API

```c++
using namespace uTensor;

const uint8_t s_a[4] = {1, 2, 3, 4};
const uint8_t s_b[4] = {5, 6, 7, 8};
const uint8_t s_c_ref[4] = {19, 22, 43, 50};

// This function  name doesnt matter, it should just be called before instantiating a model
void uTensor_init() {
  // Tell the uTensor context which allocators to use
  localCircularArenaAllocator<256> meta_allocator; // All tensor metadata gets stored here automatically, even when new is called
  localCircularArenaAllocator<256> ram_allocator;  // All temporary storage gets allocated here
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

}

void foo() {

  // Tensors are simply handles for accessing data as necessary, they are no larger than a pointer
  // RomTensor(TensorShape, data_type, data*);
  Tensor a = new /*const*/ RomTensor({2, 2}, u8, s_a);
  Tensor b = new /*const*/ RomTensor({2, 2}, u8, s_b);
  Tensor c_ref = new RomTensor({2,2}, u8, s_c_ref);
  // RamTensors are held internally and can be moved or cleared depending on the memory schedule (optional)
  Tensor c = new RamTensor({2, 2}, u8);

  // Operators take in a fixed size map of (input_name -> parameter), this gives compile time errors on input mismatching
  // Also, the name binding + lack of parameter ordering makes ctag jumping and GDB sessions significantly more intuitive
  MatrixMultOperator<uint8_t> mult_AB;
  mult_AB
      .set_inputs({{MatrixMultOperator<uint8_t>::a, a}, {MatrixMultOperator<uint8_t>::b, b}})
      .set_outputs({{MatrixMultOperator<uint8_t>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int i = 0; i < c_shape[0]; i++) {
    for (int j = 0; j < c_shape[1]; j++) {
      // Just need to cast the access to the expected type
      if( static_cast<uint8_t>(c(i, j)) != static_cast<uint8_t>(c_ref(i, j)) ) {
        printf("Oh crap!\n");
        exit(-1);
      }
    }
  }
}
```

## Building and testing locally

```
git clone git@github.com:uTensor/uTensor.git
cd uTensor/
git checkout proposal/rearch
git submodule init
git submodule update
mkdir build
cd build/
cmake -DPACKAGE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
make
make test
```


## Building and running on Arm systems
TODO


## Further Reading
- [Why Edge Computing](https://towardsdatascience.com/why-machine-learning-on-the-edge-92fac32105e6)
- [Why the Future of Machine Learning is Tiny](https://petewarden.com/2018/06/11/why-the-future-of-machine-learning-is-tiny/)
- [TensorFlow](https://www.tensorflow.org)
- [Mbed](https://developer.mbed.org)
- [Node-Viewer](https://github.com/neil-tan/tf-node-viewer/)
- [How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
- [mxnet Handwritten Digit Recognition](https://mxnet.incubator.apache.org/tutorials/python/mnist.html)
