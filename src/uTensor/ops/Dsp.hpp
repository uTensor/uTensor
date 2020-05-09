#ifndef UTENSOR_DSP_OPS_H
#define UTENSOR_DSP_OPS_H
#include "operatorBase.hpp"

// Based on MFCC op from
// https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Source/MFCC/mfcc.h

namespace uTensor {
namespace ReferenceOperators {

#define SAMP_FREQ 16000
#define NUM_FBANK_BINS 40
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000
#define M_2PI 6.283185307179586476925286766559005

template <typename T>
void convolution_kernel() {}

// This MFCC operator allocates all the tensors required for use on
// construction, then releases on destruction.
template <typename Tin, typename Tout>
class FixedMfccOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };
  FixedMfccOperator(int frame_len, int num_mfcc_features, int mfcc_dec_bits);

  ~FixedMfccOperator();

 private:
  float** create_mel_fbank();

  void populate_dct_matrix(int32_t input_length, int32_t coefficient_count);

  static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
  }

  static inline float MelScale(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
  }

 protected:
  virtual void compute() {}

 private:
  int frame_len;
  int num_mfcc_features;
  int frame_len_padded;
  int mfcc_dec_bits;
  Tensor frame;               // float
  Tensor buffer;              // float
  Tensor mel_energies;        // float
  Tensor window_func;         // float
  Tensor fbank_filter_first;  // int 32
  Tensor fbank_filter_last;   // int 32
  float** mel_fbank;
  Tensor dct_matrix;
};

template <typename Tin, typename Tout>
FixedMfccOperator<Tin, Tout>::FixedMfccOperator(int frame_len,
                                                int num_mfcc_features,
                                                int mfcc_dec_bits)
    : frame_len(frame_len),
      num_mfcc_features(num_mfcc_features),
      frame_len_padded(pow(2, ceil((log(frame_len) / log(2))))),
      mfcc_dec_bits(mfcc_dec_bits),
      frame(new RamTensor({frame_len_padded}, flt)),
      buffer(new RamTensor({frame_len_padded}, flt)),
      // mel_energies(new RamTensor({NUM_FBANK_BINS}, flt)),
      window_func(new RamTensor({frame_len}, flt)),
      fbank_filter_first(new RamTensor({NUM_FBANK_BINS}, i32)),
      fbank_filter_last(new RamTensor({NUM_FBANK_BINS}, i32)),
      dct_matrix(new RamTensor({NUM_FBANK_BINS * num_mfcc_features}, flt)) {
  for (int i = 0; i < frame_len; i++)
    window_func(i) = 0.5 - 0.5 * cos(M_2PI * ((float)i) / (frame_len));

  mel_fbank = create_mel_fbank();
  populate_dct_matrix(NUM_FBANK_BINS, num_mfcc_features);
}

template <typename Tin, typename Tout>
FixedMfccOperator<Tin, Tout>::~FixedMfccOperator() {
  for (int i = 0; i < NUM_FBANK_BINS; i++) delete mel_fbank[i];
  delete mel_fbank;
}

template <typename Tin, typename Tout>
float** FixedMfccOperator<Tin, Tout>::create_mel_fbank() {
  int32_t bin, i;

  int32_t num_fft_bins = frame_len_padded / 2;
  float fft_bin_width = ((float)SAMP_FREQ) / frame_len_padded;
  float mel_low_freq = MelScale(MEL_LOW_FREQ);
  float mel_high_freq = MelScale(MEL_HIGH_FREQ);
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS + 1);

  float* this_bin = new float[num_fft_bins];

  float** mel_fbank = new float*[NUM_FBANK_BINS];

  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    float left_mel = mel_low_freq + bin * mel_freq_delta;
    float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
    float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    int32_t first_index = -1, last_index = -1;

    for (i = 0; i < num_fft_bins; i++) {
      float freq = (fft_bin_width * i);  // center freq of this fft bin.
      float mel = MelScale(freq);
      this_bin[i] = 0.0;

      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel - mel) / (right_mel - center_mel);
        }
        this_bin[i] = weight;
        if (first_index == -1) first_index = i;
        last_index = i;
      }
    }

    fbank_filter_first(bin) = first_index;
    fbank_filter_last(bin) = last_index;
    mel_fbank[bin] = new float[last_index - first_index + 1];

    int32_t j = 0;
    // copy the part we care about
    for (i = first_index; i <= last_index; i++) {
      mel_fbank[bin][j++] = this_bin[i];
    }
  }
  delete[] this_bin;
  return mel_fbank;
}

template <typename Tin, typename Tout>
void FixedMfccOperator<Tin, Tout>::populate_dct_matrix(
    int32_t input_length, int32_t coefficient_count) {
  int32_t k, n;
  float normalizer;
  arm_sqrt_f32(2.0 / (float)input_length, &normalizer);
  for (k = 0; k < coefficient_count; k++) {
    for (n = 0; n < input_length; n++) {
      dct_matrix(k * input_length + n) =
          normalizer * cos(((double)M_PI) / input_length * (n + 0.5) * k);
    }
  }
}
}
}  // namespace uTensor
#endif
