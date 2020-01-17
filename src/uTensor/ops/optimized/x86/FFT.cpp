#include "FFT.hpp"
#include "fftw3.h"

namespace uTensor {
void dummy() {
//     fftw_real in[N], out[N], power_spectrum[N/2+1];
//     rfftw_plan p;
//     int k;
//     p = rfftw_create_plan(N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
//     rfftw_one(p, in, out);
//     power_spectrum[0] = out[0]*out[0];  /* DC component */
//     for (k = 1; k < (N+1)/2; ++k)  /* (k < N/2 rounded up) */
//          power_spectrum[k] = out[k]*out[k] + out[N-k]*out[N-k];
//     if (N % 2 == 0) /* N is even */
//          power_spectrum[N/2] = out[N/2]*out[N/2];  /* Nyquist freq. */
//     rfftw_destroy_plan(p);
}

}
