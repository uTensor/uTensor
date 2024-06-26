/* Auto-generated by utensor cli */
#include "uTensor.h"
#include "tanh_model.hpp"
#include "params_tanh_model.hpp"


TanhModel::TanhModel () :
op_TanhOperator_000()
, op_DequantizeOperator_001()
, op_QuantizeOperator_002()
{
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&metadata_allocator);
  // TODO: moving ROMTensor declarations here
}

void TanhModel::compute()
{
  // update context in case there are multiple models being run
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&metadata_allocator);
  // start rendering local snippets
  Tensor t_input_1_int80 = new RamTensor({ 128 }, i8);
    int32_t t_input_1_int80_zp = -128;
    float t_input_1_int80_scale = 0.0039215405;
    PerTensorQuantizationParams t_input_1_int80_quant_params(t_input_1_int80_zp, t_input_1_int80_scale);
    t_input_1_int80->set_quantization_params(t_input_1_int80_quant_params);


  op_QuantizeOperator_002
    .set_inputs({
        { TflmSymQuantOps::QuantizeOperator<int8_t, float>::input, inputs[input_0].tensor() },
    })
    .set_outputs({
        { TflmSymQuantOps::QuantizeOperator<int8_t, float>::output, t_input_1_int80}
    })
    .eval();

  Tensor t_Identity_int80 = new RamTensor({ 128 }, i8);
    int32_t t_Identity_int80_zp = 0;
    float t_Identity_int80_scale = 0.0078125;
    PerTensorQuantizationParams t_Identity_int80_quant_params(t_Identity_int80_zp, t_Identity_int80_scale);
    t_Identity_int80->set_quantization_params(t_Identity_int80_quant_params);


  op_TanhOperator_000
    .set_inputs({
        { ReferenceOperators::TanhOperator<int8_t, int8_t>::act_in, t_input_1_int80 },
    })
    .set_outputs({
        { ReferenceOperators::TanhOperator<int8_t, int8_t>::act_out, t_Identity_int80}
    })
    .eval();

  t_input_1_int80.free();

  op_DequantizeOperator_001
    .set_inputs({
        { TflmSymQuantOps::DequantizeOperator<float, int8_t>::a, t_Identity_int80 },
    })
    .set_outputs({
        { TflmSymQuantOps::DequantizeOperator<float, int8_t>::b, outputs[output_0].tensor()}
    })
    .eval();

  t_Identity_int80.free();
  // end of rendering local snippets
}