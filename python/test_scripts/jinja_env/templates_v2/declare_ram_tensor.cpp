Tensor {{tensor.name}} = new RamTensor({ {%for s in tensor.shape%}{{ s }}{{"," if not loop.last}}{%endfor%} }, {{ tensor.utype }});
{%if tensor.is_quantized() %}
  {% if tensor.per_tensor_quantization %}
  {{tensor.name}}->set_quantization_params(PerTensorQuantizationParams({{tensor.quantize_params.ref_zp}}, {{tensor.quantize_params.ref_scale}}));
  {% else %}
  {{tensor.name}}->set_quantization_params(PerChannelQuantizationParams({{tensor.quantize_params.ref_zp}}, {{tensor.quantize_params.ref_scale}}));
  {% endif %}
{%endif%}
