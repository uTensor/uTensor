#ifndef __UTENSOR_H
#define __UTENSOR_H
/*
 * Core bits
 */
// This one selects platform specific stuff. probably should come first
#include "uTensor/core/context.hpp"
#include "uTensor/core/errorHandler.hpp"
#include "uTensor/core/memoryManagementInterface.hpp"
#include "uTensor/core/modelBase.hpp"
#include "uTensor/core/operatorBase.hpp"
#include "uTensor/core/quantizationPrimitives.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/tensorBase.hpp"
#include "uTensor/core/TensorMap.hpp"
#include "uTensor/core/types.hpp"
#include "uTensor/core/utensor_string.hpp"
#include "uTensor/core/uTensor_util.hpp"

/*
 * Allocators
 */
#include "uTensor/allocators/arenaAllocator.hpp"

/*
 * Operators
 */
#include "uTensor/ops/ActivationFncs.hpp"
#include "uTensor/ops/ArgMinMax.hpp"
#include "uTensor/ops/Arithmetic.hpp"
#include "uTensor/ops/Functional.hpp"
#include "uTensor/ops/Convolution.hpp"
#include "uTensor/ops/Matrix.hpp"
#include "uTensor/ops/symmetric_quantization/depthwise_separable_convolution.hpp"
#include "uTensor/ops/symmetric_quantization/QuantizeOps.hpp"
#include "uTensor/ops/symmetric_quantization/fully_connected.hpp"
#include "uTensor/ops/Reshape.hpp"

/*
 * Tensors
 */

#include "uTensor/tensors/BufferTensor.hpp"
#include "uTensor/tensors/RamTensor.hpp"
#include "uTensor/tensors/RomTensor.hpp"

/*
 * Error Handlers
 */
#include "uTensor/errorHandlers/SimpleErrorHandler.hpp"

#endif
