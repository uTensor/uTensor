#include <stdlib.h>

#include <iostream>
#include <random>

#include "params_tanh_model.hpp"
#include "tanh_model.hpp"
#include "uTensor.h"

using std::cout;
using std::endl;
using uTensor::RamTensor;
using uTensor::RomTensor;
using uTensor::Tensor;

std::random_device rd;
std::mt19937 rengine(rd());
std::uniform_real_distribution<float> dist(0, 1);
float inputs_data[128];

int main(int argc, const char** argv) {
  TanhModel tanh_model;
  for (size_t i = 0; i < 128; ++i) {
    inputs_data[i] = dist(rengine);
  }
  Tensor in_tensor = new RomTensor({128}, inputs_data);
  Tensor out_tensor = new RamTensor({128}, flt);
  tanh_model.set_inputs({{TanhModel::input_0, in_tensor}})
      .set_outputs({{TanhModel::output_0, out_tensor}})
      .eval();
  for (size_t i = 0; i < 128; ++i) {
    cout << static_cast<float>(in_tensor(i)) << " -> "
         << static_cast<float>(out_tensor(i)) << endl;
  }
  return 0;
}