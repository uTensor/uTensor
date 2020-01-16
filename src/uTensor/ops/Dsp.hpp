#ifndef UTENSOR_DSP_OPS_H
#define UTENSOR_DSP_OPS_H
#include "operatorBase.hpp"

// Based on MFCC op from https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Source/MFCC/mfcc.h

namespace uTensor {


#define SAMP_FREQ 16000
#define NUM_FBANK_BINS 40
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000
#define M_2PI 6.283185307179586476925286766559005

template<typename T>
void convolution_kernel(){
}

template<typename Tin, typename Tout>
class MfccOperator : public OperatorInterface<1, 1> {
public:
    enum names: uint8_t {out, in};
    MfccOperator(int frame_len, int num_mfcc_features, int mfcc_dec_bits) : 
      frame_len(frame_len),
      num_mfcc_features(num_mfcc_features),
      frame_len_padded( pow(2,ceil((log(frame_len)/log(2)))) ),
      mfcc_dec_bits(mfcc_dec_bits),
      frame(new RamTensor({frame_len_padded}, flt)),
      buffer(new RamTensor({frame_len_padded}, flt)),
      mel_energies(new RamTensor({NUM_FBANK_BINS}, flt)),
      window_func(new RamTensor({frame_len}, flt)),
      fbank_filter_first(new RamTensor({NUM_FBANK_BINS}, i32)),
      dct_matrix(new RamTensor({NUM_FBANK_BINS * num_mfcc_features}, flt)),
      mel_fbank(new RamTensor({NUM_FBANK_BINS, frame_len_padded/2}, flt))
    {
    }

protected:
    virtual void compute() {
        convolution_kernel<T>(
            *outputs[out].tensor, 
            *inputs[in].tensor, 
            *inputs[filter].tensor,
            _padding,
            _stride
            );
    }
private:
    int frame_len;
    int num_mfcc_features;
    int frame_len_padded;
    int mfcc_dec_bits;
    Tensor frame;                // float 
    Tensor buffer;               // float
    Tensor mel_energies;         // float
    Tensor window_func;          // float 
    Tensor fbank_filter_first; //int 32
    Tensor fbank_filter_last; // int 32
    float ** mel_fbank;
    float * dct_matrix;
};

}
#endif
