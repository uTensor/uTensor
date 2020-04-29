Tensor {{tensor_name}} = new RomTensor({ {%for s in shape%}{{ s }}{{"," if not loop.last}}{%endfor%} }, {{TENSOR_TYPE_MAP[tensor_type_str]}}, {{const_var_name}});
{%if quantize_params%}
  {{tensor_name}}->set_quantization_params(PerTensorQuantizationParams({{quantize_params[1]}}, {{quantize_params[0]}}));
{%endif%}