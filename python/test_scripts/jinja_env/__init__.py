import jinja2
from pathlib import Path

_template_dir = Path(__file__).parent / "templates"

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_template_dir), trim_blocks=True, lstrip_blocks=True
)
env.globals.update(
    zip=zip,
    len=len,
    TENSOR_TYPE_MAP={
        "int8_t": "i8",
        "uint8_t": "u8",
        "int16_t": "i16",
        "uint16_t": "u16",
        "int32_t": "i32",
        "uint32_t": "u32",
        "float": "flt",
    },
)

del _template_dir
