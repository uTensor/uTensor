{% for x in arr %}{{ x }}{{ "," if not loop.last }}{% endfor %}
