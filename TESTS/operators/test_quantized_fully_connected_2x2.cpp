
#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_quantized_fully_connected2x2.hpp"
using std::cout;
using std::endl;

using namespace uTensor;



TEST(Quantized, reference_3_fully_connected) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2*2*sizeof(i8), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  
  Tensor A =  new RomTensor({ 1, 2 }, i8, s_ref_A);
                        A->set_quantization_params(PerTensorQuantizationParams(s_ref_A_zp[0], s_ref_A_scale[0]));

  
  Tensor filter =  new RomTensor({ 2, 2 }, i8, s_ref_filter);
                        filter->set_quantization_params(PerTensorQuantizationParams(s_ref_filter_zp[0], s_ref_filter_scale[0]));

  
  Tensor bias =  new RomTensor({ 2 }, i32, s_ref_bias);
                        bias->set_quantization_params(PerTensorQuantizationParams(s_ref_bias_zp[0], s_ref_bias_scale[0]));

  
  Tensor output_ref =  new RomTensor({ 1, 2 }, i8, s_ref_output_ref);
                        output_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_output_ref_zp[0], s_ref_output_ref_scale[0]));

  
  Tensor out = new RamTensor({ 1, 2 }, i8);
                out->set_quantization_params(PerTensorQuantizationParams(s_ref_output_ref_zp[0], s_ref_output_ref_scale[0] ));

  QuantizedFullyConnectedOperator<int8_t> op(
      TFLM::TfLiteFusedActivation::kTfLiteActRelu);
  op.set_inputs({{QuantizedFullyConnectedOperator<int8_t>::input, A},
                 {QuantizedFullyConnectedOperator<int8_t>::filter, filter},
                 {QuantizedFullyConnectedOperator<int8_t>::bias, bias}})
      .set_outputs({{QuantizedFullyConnectedOperator<int8_t>::output, out}})
      .eval();

  for (int i = 0; i < out->num_elems(); ++i) {
    int8_t out_value = (int8_t) out(i);
    EXPECT_NEAR(out_value, s_ref_output_ref[i], 5);
  }
}
