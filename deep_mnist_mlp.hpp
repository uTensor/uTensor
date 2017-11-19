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

void tensorQuantize(Context& ctx, TName input, TName output,
  TName out_min, TName out_max); 

void ReluLayer(Context& ctx, TName x, TName x_min, TName x_max,
   TName w, TName w_min, TName w_max, TName b,
    TName z_output); 

void PredLayer(Context &ctx, TName input, TName input_min,
               TName input_max, TName output, TName w, TName w_min, TName w_max, TName bias, TName dim);

int runMLP(string inputIdxFile);

#endif
