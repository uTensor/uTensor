#include "gemmlowp.hpp"

#include <numeric>

#include "context.hpp"

using namespace std;

namespace uTensor {
namespace gemmlowp {

DEFINE_ERROR(InvalidExponentError)

int32_t MaskIfNonZero(int32_t a) {
  static constexpr int32_t zero = 0;
  return a ? ~zero : zero;
}

int32_t MaskIfLessThan(int32_t a, int32_t b) { return MaskIfNonZero(a < b); }

int32_t MaskIfGreaterThan(int32_t a, int32_t b) { return MaskIfNonZero(a > b); }

int32_t RoundingDivideByPOT(int32_t x, int32_t exponent) {
  if (exponent > 31 || exponent < 0) {
    Context::get_default_context()->throwError(new InvalidExponentError);
  }
  const int32_t mask = (1ll << exponent) - 1;
  const int32_t zero = 0;
  const int32_t one = 1;
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + (MaskIfLessThan(x, zero) & one);
  return (x >> exponent) + (MaskIfGreaterThan(remainder, threshold) & one);
}

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

}  // namespace gemmlowp
}  // namespace uTensor