#include <iostream>
#include "gtest/gtest.h"
#include "uTensor.h"

using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

TEST(QuantizationParams, base_params) {
  bool result = false;
  mErrHandler.set_onError([&result](Error* err){
    if(*err == AttemptToUseUnSpecializedQuantizeParamsError()) {
      result = true; 
    }
  });

  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  QuantizationParams mParams;
  testing::internal::CaptureStdout(); // Dont print error messages since they will be confusing
  int32_t i = mParams.get_zeroP_for_channel(0);
  EXPECT_EQ(result, true);
  EXPECT_EQ(i, 0);
  
  result = false;
  float f = mParams.get_scale_for_channel(0);
  EXPECT_EQ(result, true);
  EXPECT_EQ(f, 0.0f);
}
