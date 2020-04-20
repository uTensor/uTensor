{%if shape %}
Tensor {{tensor_name}} = new RamTensor({ {%for s in shape%}{{ s }}{{"," if not loop.last}}{%endfor%} }, {{TENSOR_TYPE_MAP[tensor_type_str]}});
{% else %}
Tensor {{tensor_name}} = new RamTensor({{TENSOR_TYPE_MAP[tensor_type_str]}});
{%endif%}