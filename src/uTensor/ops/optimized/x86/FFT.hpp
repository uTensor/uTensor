#ifndef UTENSOR_FFT_HPP
#define UTENSOR_FFT_HPP

#include "fftw3.h"
#include "operatorBase.hpp"

namespace uTensor {
  //fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned flags);

void power_spectrum_kernel(Tensor& power, const Tensor& input);

class PowerSpectrum : public OperatorInterface<1,1> {
  public:
    enum names: uint8_t { input, power };

  protected:
    virtual void compute() {

    }
};
}
#endif
