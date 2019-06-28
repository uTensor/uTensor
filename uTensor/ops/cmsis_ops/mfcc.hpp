#ifndef UTENSOR_KWS_MFCC_FUNCTIONS
#define UTENSOR_KWS_MFCC_FUNCTIONS
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensorBase.hpp"

/**
 * @param [in] data input tensor
 */
template <typename TIn, typename TOut>
void MfccCmsis<Tin, Tout>(S_TENSOR input_tensor, S_TENSOR mel_fbank_tensor, S_TENSOR window_func_tensor,
                          S_TENSOR dct_matrix_tensor, S_TENSOR fbank_filter_first_tensor, S_TENSOR fbank_filter_last_tensor,
                          int frame_len, int num_mfcc_features int num_fbank_bins, S_TENSOR mfcc_out_tensor, Tin dummy1, Tout dummy2) {
    printf("Error not supported in the general case\n");
    exit(-1);
}

template <>
void MfccCmsis<float, int7_t>(S_TENSOR input_tensor, S_TENSOR mel_fbank_tensor, S_TENSOR window_func_tensor,
                              S_TENSOR dct_matrix_tensor, S_TENSOR fbank_filter_first_tensor, S_TENSOR fbank_filter_last_tensor,
                              int frame_len, int num_mfcc_features int num_fbank_bins, S_TENSOR mfcc_out_tensor, Tin dummy1, Tout dummy2) {
    //Implementation for MFCC
    int32_t i, j, bin;
    int32_t frame_len_padded;

    //Buffers
    frame_len_padded = pow(2, ceil((log(frame_len) / log(2))));
    RamTensor<float>* frame_tensor = new RamTensor<float>({ frame_len_padded });
    RamTensor<float>* buffer_tensor = new RamTensor<float>({ frame_len_padded });
    RamTensor<float>* mel_energies_tensor = new RamTensor<float>({ num_fbank_bins });
    float* frame = frame_tensor->write<float>(0, 0);
    float* buffer = buffer_tensor->write<float>(0, 0);
    float* mel_energies = mel_energies_tensor->write<float>(0, 0);

    //Inputs
    float* input_data = input_tensor->read<float>(0, 0);
    float* mel_fbank = mel_fbank_tensor->read<float>(0, 0);
    float* window_func = window_func_tensor->read<float>(0, 0);
    float* dct_matrix = dct_matrix_tensor->read<float>(0, 0);
    float* fbank_filter_first = fbank_filter_first_tensor->read<float>(0, 0);
    float* fbank_filter_last = fbank_filter_last_tensor->read<float>(0, 0);

    //Output
    float* mfcc_out = mfcc_out_tensor->write<float>(0, 0);

    //TensorFlow way of normalizing .wav data to (-1,1)
    for (i = 0; i < frame_len; i++) {
        frame[i] = (float)input_data[i]/(1<<15); 
    }
    //Fill up remaining with zeros
    memset(&frame[frame_len], 0, sizeof(float) * (frame_len_padded-frame_len));

    for (i = 0; i < frame_len; i++) {
        frame[i] *= window_func[i];
    }

    //Compute FFT
    arm_rfft_fast_f32(rfft, frame, buffer, 0);

    //Convert to power spectrum
    //frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
    int32_t half_dim = frame_len_padded/2;
    float first_energy = buffer[0] * buffer[0],
          last_energy =  buffer[1] * buffer[1];  // handle this special case
    for (i = 1; i < half_dim; i++) {
        float real = buffer[i*2], im = buffer[i*2 + 1];
        buffer[i] = real*real + im*im;
    }
    buffer[0] = first_energy;
    buffer[half_dim] = last_energy;  
 
    float sqrt_data;
    //Apply mel filterbanks
    for (bin = 0; bin < num_fbank_bins; bin++) {
        j = 0;
        float mel_energy = 0;
        int32_t first_index = fbank_filter_first[bin];
        int32_t last_index = fbank_filter_last[bin];
        for (i = first_index; i <= last_index; i++) {
            arm_sqrt_f32(buffer[i],&sqrt_data);
            mel_energy += (sqrt_data) * mel_fbank[bin][j++];
        }
        mel_energies[bin] = mel_energy;

        //avoid log of zero
        if (mel_energy == 0.0)
            mel_energies[bin] = FLT_MIN;
    }

    //Take log.
    for (bin = 0; bin < num_fbank_bins; bin++)
        mel_energies[bin] = logf(mel_energies[bin]);

    //Take DCT. Uses matrix mul.
    for (i = 0; i < num_mfcc_features; i++) {
        float sum = 0.0;
        for (j = 0; j < num_fbank_bins; j++) {
            sum += dct_matrix[i*num_fbank_bins+j] * mel_energies[j];
        }

        //Input is Qx.mfcc_dec_bits (from quantization step)
        sum *= (0x1 << mfcc_dec_bits);
        sum = round(sum); 
        if(sum >= 127)
            mfcc_out[i] = 127;
        else if(sum <= -128)
            mfcc_out[i] = -128;
        else
            mfcc_out[i] = sum;
    }

    //Clean up whatever is allocated
    delete frame_tensor;
    delete buffer_tensor;
    delete mel_energies_tensor;
}

template <class Tin, class Tout>
class MfccCmsisOp : public Operator {
  public:
  MfccCmsisOp() {
    n_inputs = 8;
    n_outputs = 1;
  }
  virtual void compute() override {
      Tin x;
      Tout y;
      MfccCmsis(inputs[0], inputs[1], inputs[2], inputs[3],
                inputs[4], inputs[5], inputs[6], inputs[7], outputs[0], x, y);
  }
};

#endif 
