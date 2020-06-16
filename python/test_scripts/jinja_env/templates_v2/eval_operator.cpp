{{op.name}}
  .set_inputs({ 
    {% for x in op.input_map %}
    { {{op.type_signature}}::{{x}}, {{op.input_map[x].name}} }{{"," if not loop.last}}
  {% endfor %}
  }).set_outputs({ 
    {% for x in op.output_map %}
    { {{op.type_signature}}::{{x}}, {{op.output_map[x].name}} }{{"," if not loop.last}}
  {% endfor %}
  }).eval();
