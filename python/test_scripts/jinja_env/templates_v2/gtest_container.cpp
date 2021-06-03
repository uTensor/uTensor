#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "{{ constants_header }}"
using std::cout;
using std::endl;

using namespace uTensor;
{% for using_directive in using_directives %}
{{ using_directive }};
{% endfor %}

SimpleErrorHandler mErrHandler(10);

{% for test in tests %}
/***************************************
 * Generated Test {{loop.index}}
 ***************************************/
{{ test }}

{% endfor %}
