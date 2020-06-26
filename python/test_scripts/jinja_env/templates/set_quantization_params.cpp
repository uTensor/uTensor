  {{ qp.tensor.name }}->set_quantization_params({{ qp.quantization_type }}({{ qp.ref_zp }}, {{ qp.ref_scale }}));
