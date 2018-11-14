# uTensor Ops Porting Guide

## Setup

### Requirements
### Instructions

## C++ Operator Class
  You need to implement your operator in C++ runtime and make sure it is tested.

### Implementation

### Test Case

## Code Generator
The code generator is written in Python. There are a few places you will need to extend the code generator to support your newly created operator:

- Op Factory
- Code Snippet Handler
- Template

### Op Factory
  During uTensor graph building, the op factory is utilized by the parser to lookup the operations in the input model and create the cooresponding operator objects. A operator class can contain:

- Input and output tensor names
- Data types 
- Enable/Disablel eager/dynamic execution 
- Reference counter and other properties

In operators.py, add your oparator name into the lookup table. In this case `Uint8Q7OriginOp`:

```
  _operators = {"Add": _AddOperator,
                "ArgMax": _ArgMaxOperator,
                "Dequantize": _DequantizeOperator,
                "Max": _MaxOperator,
				...
                "Uint8Q7OriginOp", : _Uint8Q7OriginOp}
```

Now, the operator class named `_Uint8Q7OriginOp` will be instantized everytime an `Uint8Q7OriginOp` is encountered during graph parsing. Let's implement the operator function:

```
class _Uint8Q7OriginOperator(_Operator):
  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    output = op_info.output_tensors[0].name
    #tf_dtype = op_info.input_tensors[0].dtype
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE, 
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    to_eval = parser.get('to_eval', False)
    self._snippet = Uint8Q7OriginSnippet(inputs, output, ref_count, to_eval)
``` 
### Snippet Handler
In _snippets.py, we feed the information from the operator class into the template engine to generate C++ output. It includes template file location, variable mapping and header files to include.

```
class Uint8Q7OriginSnippet(Snippet):
  __template_name__ = "snippets/cmsis_uint8q7origin_op.cpp"
  __headers__ = set(['"uTensor/ops/cmsis_ops/Uint8Q7OriginOps.hpp"'])

  def __init__(self, inputs, output,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["to_eval"] = to_eval
```
Register it with the package:

```
__all__ = ["Snippet", "SnippetContainerBase",
			...
"QuantRangeForMultiplicationSnippet", "Uint8Q7OriginSnippet"]
```

### Creating the Source Template
Create a Jinja2 template file under the location specified by the `__template_name__` in the previus step.

```
{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    {% if ref_count %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}");
    {% endif %}
    ctx.push(new Uint8Q7OriginOp(),
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" }, 
             { "{{output}}" });
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}
```

Notice the variable used in the template above is consistent with the names in the snippet handler.
