#ifndef __DEEP_MNIST_MLP_HPP__
#define __DEEP_MNIST_MLP_HPP__
#include "uTensor/core/tensor.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/ops/MathOps.hpp"
#include "uTensor/ops/MatrixOps.hpp"
#include "uTensor/ops/NnOps.hpp"
#include "uTensor/ops/ArrayOps.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include "uTensor/core/context.hpp"
#include "mbed.h"

void tensorQuantize(Context& ctx, TName input, TName output,
  TName out_min, TName out_max); 

void ReluLayer(Context& ctx, TName x, TName x_min, TName x_max,
   TName w, TName w_min, TName w_max, TName b,
    TName z_output); 

void PredLayer(Context &ctx, TName input, TName input_min,
               TName input_max, TName output, TName w, TName w_min, TName w_max, TName bias, TName dim);

int runMLP(const char* inputIdxFile);

#endif
