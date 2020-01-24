#include "FFT.hpp"
#include "context.hpp"

//fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned flags);
namespace uTensor {

PowerSpectrum::PowerSpectrum(int N) : N(N) {}
void PowerSpectrum::compute() {
  power_spectrum_kernel(*outputs[power].tensor, *inputs[input].tensor, N);
}

// TODO move the plan bits out
// Potential recreate plan if needed
void PowerSpectrum::power_spectrum_kernel(Tensor& power, const Tensor& input, int N) {
  if(power->get_shape().get_linear_size() < N) {
    Context::get_default_context()->throwError(new InvalidTensorOutputError);
  }
  if(input->get_shape().get_linear_size() < N) {
    Context::get_default_context()->throwError(new InvalidTensorInputError);
  }

  float* inputf, *output;
  fftwf_complex *tmp;
  fftwf_plan p;

  uint32_t num_elems = (N/2+1);
  //Tensor tmp = new RamTensor({num_elems*2}, flt);
  //size_t size = get_writeable_block(tmp, tmpc, (uint16_t) num_elems*2, 0);
  size_t size;
  tmp = reinterpret_cast<fftwf_complex*>(Context::get_default_context()->get_ram_data_allocator()->allocate(num_elems*sizeof(fftwf_complex)));
  if(!tmp) {
    //Probably a memory allocation error
    Context::get_default_context()->throwError(new OutOfMemError);
  }
  float* tmpc = reinterpret_cast<float*>(tmp);
  size = get_readable_block(input, inputf, (uint16_t) N, 0);
  size = get_readable_block(power, output, (uint16_t) N, 0);
  //in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
  //out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
  p = fftwf_plan_dft_r2c_1d(N, inputf, tmp, FFTW_ESTIMATE);
  fftwf_execute(p); /* repeat as needed */
  fftwf_destroy_plan(p);

  for(int i = 0; i < num_elems; i++){
    float real = tmpc[i*2], im = tmpc[i*2 + 1];
    output[i] = real*real + im*im;
  }
}

}
