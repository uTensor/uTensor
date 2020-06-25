{% if tensor.ref_name %}
static const {{ tensor.dtype }} {{ tensor.ref_name }}[{{ len(tensor.flatten()) }}] = { 
  {% for x in tensor.flatten() %} {{ x }}{{ "," if not loop.last }}{{ "\n" if not loop.first and loop.index % 10 == 0}} {% endfor %} 
};
{% endif %}
{% if tensor.is_quantized() %}
  {% if tensor.per_tensor_quantization %}
static const int32_t {{ tensor.quantize_params.ref_zp }} = {{ tensor.quantize_params.zp[0] }};
static const float {{ tensor.quantize_params.ref_scale }} = {{ tensor.quantize_params.scale[0] }};
  {% else %}
static const int32_t {{ tensor.quantize_params.ref_zp }}[ {{ len(tensor.quantize_params.zp) }} ] = { 
  {% for x in tensor.quantize_params.zp %} {{ x }}{{ "," if not loop.last }}{{ "\n" if not loop.first and loop.index % 10 == 0}} {% endfor %} 
};
static const float {{ tensor.quantize_params.ref_scale }}[ {{ len(tensor.quantize_params.scale) }} ] = { 
  {% for x in tensor.quantize_params.scale %} {{ x }}{{ "," if not loop.last }}{{ "\n" if not loop.first and loop.index % 10 == 0}} {% endfor %} 
};
  {%endif %}
{% endif %}
