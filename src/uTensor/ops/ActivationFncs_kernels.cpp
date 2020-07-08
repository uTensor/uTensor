#include "ActivationFncs_kernels.hpp"

namespace uTensor {

void sq_softmax_k(Tensor& out, const Tensor& in, int8_t beta) {
  const float beta_f = static_cast<float>(beta);
  const TensorShape& inShape = in->get_shape();
  int outer_dim = inShape.num_dims() -1;
  int depth = inShape[outer_dim];
  int out_side_numelems = 1;
  for(int i = 0; i < inShape.num_dims(); i++){
    out_side_numelems *= (i == outer_dim) ? 1: inShape[i];
  }

  for (int i = 0; i < out_side_numelems; i++) {
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    float max = static_cast<float>(std::numeric_limits<int8_t>::lowest());
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<float>(static_cast<int8_t>(in(i, j))));
    }

    float mSum = 0;
    for(int j = 0; j < depth; j++){
      const int32_t in32 =  static_cast<int8_t>(in(i,j));
      const float in_scale = in->get_quantization_params().get_scale_for_channel(0);
      const int32_t in_zp = in->get_quantization_params().get_zeroP_for_channel(0);
      const float in_f = (in32 - in_zp)*in_scale;
      const float tmp = exp((in_f - max) * beta_f);
      mSum += tmp;
      //out(i,j) = tmp;
    }
    // TODO FIXME SLOW but mem efficient
    for(int j = 0; j < depth; j++){
      const int32_t in32 =  static_cast<int8_t>(in(i,j));
      const float in_scale = in->get_quantization_params().get_scale_for_channel(0);
      const int32_t in_zp = in->get_quantization_params().get_zeroP_for_channel(0);
      const float in_f = (in32 - in_zp)*in_scale;
      const float out_val = exp((in_f - max) * beta_f) / mSum;
      
      const float oscale = out->get_quantization_params().get_scale_for_channel(0);
      const int32_t ozp = out->get_quantization_params().get_zeroP_for_channel(0);
      const int32_t otmp = static_cast<int32_t>(out_val/oscale) + ozp;
      const int8_t out8 = (otmp < -127 ) ? -128 : (otmp > 127) ? 127 : static_cast<int8_t>(otmp);
      
      out(i, j)  = out8;
    }
  }

}

template <>
void sigmoid_k_impl<int8_t>::operator()(Tensor& out, const Tensor& in) const {
  const float one = 1;
  uint32_t t_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const int32_t in32 =  static_cast<int8_t>(in(i));
    const float in_scale = in->get_quantization_params().get_scale_for_channel(0);
    const int32_t in_zp = in->get_quantization_params().get_zeroP_for_channel(0);
    const float in_f = (in32 - in_zp)*in_scale;
    const float out_val = one / (one + exp( -in_f ));
    const float oscale = out->get_quantization_params().get_scale_for_channel(0);
    const int32_t ozp = out->get_quantization_params().get_zeroP_for_channel(0);
    const int32_t otmp = static_cast<int32_t>(out_val/oscale) + ozp;
    const int8_t out8 = (otmp < -127 ) ? -128 : (otmp > 127) ? 127 : static_cast<int8_t>(otmp);  
    out(i) = out8;
  }
}

}
