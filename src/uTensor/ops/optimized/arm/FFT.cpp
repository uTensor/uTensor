#include "FFT.hpp"
#include "context.hpp"

// fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned
// flags);
namespace uTensor {

PowerSpectrum::PowerSpectrum(int N) : N(N), rfft(new arm_rfft_fast_instance_f32) {
  arm_rfft_fast_init_f32(rfft, N);
}
PowerSpectrum::~PowerSpectrum() {
  delete rfft;
}
void PowerSpectrum::compute() {
  power_spectrum_kernel(*outputs[power].tensor, *inputs[input].tensor, N);
}

// TODO move the plan bits out
// Potential recreate plan if needed
void PowerSpectrum::power_spectrum_kernel(Tensor& power, const Tensor& input,
                                          int N) {
  if (power->get_shape().get_linear_size() < N) {
    Context::get_default_context()->throwError(new InvalidTensorOutputError);
  }
  if (input->get_shape().get_linear_size() < N) {
    Context::get_default_context()->throwError(new InvalidTensorInputError);
  }

  float *inputf, *output;

  uint32_t num_elems = (N / 2 + 1);
  // Tensor tmp = new RamTensor({num_elems*2}, flt);
  // size_t size = get_writeable_block(tmp, tmpc, (uint16_t) num_elems*2, 0);
  size_t size;
  float* tmpc = reinterpret_cast<float*>(
      Context::get_default_context()->get_ram_data_allocator()->allocate(
          N * sizeof(float)));
  if (!tmpc) {
    // Probably a memory allocation error
    Context::get_default_context()->throwError(new OutOfMemError);
  }
  //TODO uncomment this crap
//  
//  size = get_readable_block(input, inputf, (uint16_t)N, 0);
//  size = get_readable_block(power, output, (uint16_t)N, 0);
//  // in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
//  // out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
//  
//  arm_rfft_fast_f32(rfft, frame, buffer, 0);
//  p = fftwf_plan_dft_r2c_1d(N, inputf, tmpc, FFTW_ESTIMATE);
//  fftwf_execute(p); /* repeat as needed */
//  fftwf_destroy_plan(p);
//
//  float first_energy = tmpc[0] * tmpc[0],
//        last_energy =  tmpc[1] * tmpc[1];  // handle this special case
//  int half_dim = N/2;
//  for (i = 1; i < half_dim; i++) {
//    float real = tmpc[i*2], im = tmpc[i*2 + 1];
//    output[i] = real*real + im*im;
//  }
//  output[0] = first_energy;
//  output[half_dim] = last_energy;
//  
//  Context::get_default_context()->get_ram_data_allocator()->deallocate(tmpc);
}

}  // namespace uTensor
