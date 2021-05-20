{{ op.ns }}{{ op.op_type }}{% if op.dtypes %}<{{ op.array_template.render(arr=op.dtypes) }}>{% endif %}
