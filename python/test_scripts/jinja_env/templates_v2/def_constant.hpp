static const {{ tensor.dtype }} {{ tensor.ref_name }}[{{ len(tensor.flatten()) }}] = { 
  {% for x in tensor.flatten() %} {{ x }}{{ "," if not loop.last }}{{ "\n" if not loop.first and loop.index % 10 == 0}} {% endfor %} 
};
