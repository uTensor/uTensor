#ifndef TFLM_DEFS_H
#define TFLM_DEFS_H

namespace uTensor {
namespace TFLM {

typedef enum {
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActRelu1,  // min(max(-1, x), 1)
  kTfLiteActRelu6,  // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;

}
}
#endif
