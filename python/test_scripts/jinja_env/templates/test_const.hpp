#ifndef _{{test_name | upper}}_H
#define _{{test_name | upper}}_H

{%for name, (value, type_str) in constants_map.items()%}
static const {{type_str}} {{name}}[{{len(value)}}] = { {%for v in value%}{{v}}{{", " if not loop.last}}{%endfor%} };
{%endfor%}

#endif // _{{test_name | upper}}_H