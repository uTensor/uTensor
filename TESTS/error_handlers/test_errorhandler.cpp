#include <iostream>
#include "gtest/gtest.h"
#include "uTensor.h"

using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);
DECLARE_ERROR(MyDumbError);
DEFINE_ERROR(MyDumbError);

TEST(SimpleCustomErrorHandler, handle_error) {
  bool result = false;
  mErrHandler.set_onError([&result](Error* err){
    if(*err == MyDumbError()) {
      result = true; 
    }
  });

  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  Context::get_default_context()->throwError(new MyDumbError);
  EXPECT_EQ(result, true);
}
