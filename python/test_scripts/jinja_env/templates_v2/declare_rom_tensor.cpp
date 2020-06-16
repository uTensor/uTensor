Tensor {{ tensor.name }} = new RomTensor({ {% for s in tensor.shape %}{{ s }}{{"," if not loop.last}}{% endfor %} }, {{ tensor.utype }}, {{ tensor.ref_name }});
{%if tensor.quantize_params%}
  {{tensor.name}}->set_quantization_params(PerTensorQuantizationParams({{tensor.quantize_params[1]}}, {{tensor.quantize_params[0]}}));
{%endif%}
