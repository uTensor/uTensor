#ifndef __DEEP_MNIST_MLP_HPP__
#define __DEEP_MNIST_MLP_HPP__
#include "mbed.h"
#include "tensor.hpp"
#include "test.hpp"
#include "tensorIdxImporter.hpp"
#include "MathOps.hpp"
#include "MatrixOps.hpp"
#include "NnOps.hpp"
#include "ArrayOps.hpp"
#include "uTensor_util.hpp"
#include "uTensorBase.hpp"
#include "context.hpp"

void tensorQuantize(Context& ctx, TENSOR input, TENSOR output,
  TENSOR out_min, TENSOR out_max); 

void ReluLayer(Context& ctx, TENSOR x, TENSOR x_min, TENSOR x_max,
   TENSOR w, TENSOR w_min, TENSOR w_max, TENSOR b,
    TENSOR z_output); 

void PredLayer(Context &ctx, TENSOR input, TENSOR input_min,
               TENSOR input_max, TENSOR output, TENSOR w, TENSOR w_min, TENSOR w_max, TENSOR bias, TENSOR dim);

int runMLP(string inputIdxFile);

#endif
