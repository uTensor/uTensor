#ifndef _UTENSOR_GEMMLOWP
#define _UTENSOR_GEMMLOWP
#include <cstdint>

#include "uTensor/core/errorHandler.hpp"

namespace uTensor {
namespace gemmlowp {

DECLARE_ERROR(InvalidExponentError);

std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b);
std::int32_t RoundingDivideByPOT(std::int32_t x, std::int32_t exponent);

}  // namespace gemmlowp
}  // namespace uTensor
#endif
