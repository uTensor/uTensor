#include "test_helper.h"
#include "src/uTensor/loaders/tensorIdxImporter.hpp"
#include "MathOps.hpp"

#include <iostream>
using std::cout;
using std::endl;

TensorIdxImporter t_import;
Context ctx;


void test_operators_requantizationRange(void) {
  ctx.gc();

  //Note: raw pointers should be owned ONLY by the context. no copy of the raw pointer should exist elsewhere
  // reference inputs
  ctx.addCached(hold(t_import.int_import("/fs/constants/rqRange/in/qMatMul_0.idx")), "a");
  ctx.addCached(hold(t_import.float_import("/fs/constants/rqRange/in/qMatMul_1.idx")), "a_min");
  ctx.addCached(hold(t_import.float_import("/fs/constants/rqRange/in/qMatMul_2.idx")), "a_max");

  // reference output
  ctx.addCached(hold(t_import.float_import("/fs/constants/rqRange/out/rqRange_0.idx")), "ref_min");
  ctx.addCached(hold(t_import.float_import("/fs/constants/rqRange/out/rqRange_1.idx")), "ref_max");

  // Implementation goes here

  // modify the checks below:
  ctx.addCached(hold(new RamTensor<float>(ctx.get("ref_min")->getShape())), "out_min");
  ctx.addCached(hold(new RamTensor<float>(ctx.get("ref_max")->getShape())), "out_max");
  TNameList inputs = {"a", "a_min", "a_max"};
  TNameList outputs = {"out_min", "out_max"};

  ctx.push_static(hold(new Requantization_RangeOp()), "Requantization_RangeOp", inputs, outputs);
  ctx.eval();

  //Note: an output tensor will not be auto-deleted by context unless it has been used as an input
  double result =
      meanPercentErr<float>(ctx.get("ref_min").get(), ctx.get("out_min").get())
       + meanPercentErr<float>(ctx.get("ref_max").get(), ctx.get("out_max").get());
  // passed(result < 0.0001);
  EXPECT_EQ(result, 0);
}

void test_operators_requantize(void) {
  ctx.gc();

  // reference inputs
  ctx.addCached(hold(t_import.int_import("/fs/constants/rQ/in/qMatMul_0.idx")), "a");
  ctx.addCached(hold(t_import.float_import("/fs/constants/rQ/in/qMatMul_1.idx")), "a_min");
  ctx.addCached(hold(t_import.float_import("/fs/constants/rQ/in/qMatMul_2.idx")), "a_max");
  ctx.addCached(hold(t_import.float_import("/fs/constants/rQ/in/rqRange_0.idx")), "r_a_min");
  ctx.addCached(hold(t_import.float_import("/fs/constants/rQ/in/rqRange_1.idx")), "r_a_max");
  // tf.quint8

  //Note:
  //Instead of using ctx.get() to obtain a shared_ptr, you may also use the shared_ptr returned by ctx.add()

  // reference outputs
  S_TENSOR ref_a_q = ctx.addCached(hold(t_import.ubyte_import("/fs/constants/rQ/out/rQ_0.idx")), "ref_a_q");
  S_TENSOR ref_a_min = ctx.addCached(hold(t_import.float_import("/fs/constants/rQ/out/rQ_1.idx")), "ref_a_min");
  S_TENSOR ref_a_max = ctx.addCached(hold(t_import.float_import("/fs/constants/rQ/out/rQ_2.idx")), "ref_a_max");

  // modify the checks below:
  S_TENSOR a_q = ctx.addCached(hold(new RamTensor<unsigned char>(ref_a_q->getShape())), "a_q");
  S_TENSOR a_min_q = ctx.addCached(hold(new RamTensor<float>(ref_a_min->getShape())), "a_min_q");
  S_TENSOR a_max_q = ctx.addCached(hold(new RamTensor<float>(ref_a_max->getShape())), "a_max_q");


  TNameList inputs = {"a", "a_min", "a_max", "r_a_min", "r_a_max"};
  TNameList outputs = {"a_q", "a_min_q", "a_max_q"};

  // Implementation goes here
  
  ctx.push_static(hold(new RequantizeOp()), "RequantizeOp", inputs, outputs);
  ctx.eval();
  

  double result = meanPercentErr<unsigned char>(ref_a_q.get(), a_q.get()) +
                  meanPercentErr<float>(ref_a_min.get(), a_min_q.get()) +
                  meanPercentErr<float>(ref_a_max.get(), a_max_q.get());
  // passed(result < 0.0001);
  EXPECT_EQ(result, 0);
}

void test_operators_requantize2(void) {
  ctx.gc();

  // reference inputs
  ctx.addCached(hold(t_import.int_import("/fs/constants/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_0.idx")), "a");
  ctx.addCached(hold(t_import.float_import("/fs/constants/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_1.idx")), "a_min");
  ctx.addCached(hold(t_import.float_import("/fs/constants/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_2.idx")), "a_max");
  ctx.addCached(hold(t_import.float_import("/fs/constants/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_0.idx")), "r_a_min");
  ctx.addCached(hold(t_import.float_import("/fs/constants/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_1.idx")), "r_a_max");
  // tf.quint8

  // reference outputs
  ctx.addCached(hold(t_import.ubyte_import("/fs/constants/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_0.idx")), "ref_a_q");
  ctx.addCached(hold(t_import.float_import("/fs/constants/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_1.idx")), "ref_a_min");
  ctx.addCached(hold(t_import.float_import("/fs/constants/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_2.idx")), "ref_a_max");


  // modify the checks below:
  ctx.addCached(hold(new RamTensor<unsigned char>(ctx.get("ref_a_q")->getShape())), "a_q");
  ctx.addCached(hold(new RamTensor<float>(ctx.get("ref_a_min")->getShape())), "a_min_q");
  ctx.addCached(hold(new RamTensor<float>(ctx.get("ref_a_max")->getShape())), "a_max_q");

  S_TENSOR ref_val = ctx.get("ref_a_q");
  S_TENSOR ref_min = ctx.get("ref_a_min");
  S_TENSOR ref_max = ctx.get("ref_a_max");
  S_TENSOR out_val = ctx.get("a_q");
  S_TENSOR out_min = ctx.get("a_min_q");
  S_TENSOR out_max = ctx.get("a_max_q");

  // Implementation goes here
  
  ctx.push_static(hold(new RequantizeOp()), "RequantizeOp", {"a", "a_min", "a_max", "r_a_min", "r_a_max"}, {"a_q", "a_min_q", "a_max_q"});
  ctx.eval();
  

  double result;
  if((result = meanPercentErr<unsigned char>(ctx.get("ref_a_q").get(), out_val.get())) != 0) {
      printf("Requantize a_q failed (%.6f)\r\n", result);
      unsigned char* ref_ptr = ref_val.get()->write<unsigned char>(0, 0);
      unsigned char* test_ptr = out_val.get()->write<unsigned char>(0, 0);
      for(uint32_t i = 0; i < ref_val->getSize(); i++) {
          if(ref_ptr[i] != test_ptr[i]) {
              printf("%u: %d != %d\r\n", i, ref_ptr[i], test_ptr[i]);
          } else {
              printf("%u: %d == %d\r\n", i, ref_ptr[i], test_ptr[i]);
          }
      }
  }


  if((result = meanPercentErr<float>(ref_min.get(), out_min.get())) != 0) printf("Requantize a_min_q failed (%.6f)\r\n", result);

  if((result = meanPercentErr<float>(ref_max.get(), out_max.get())) != 0) printf("Requantize a_max_q failed (%.6f)\r\n", result);

  result = meanPercentErr<unsigned char>(ref_val.get(), out_val.get()) +
                  meanPercentErr<float>(ref_min.get(), out_min.get()) +
                  meanPercentErr<float>(ref_max.get(), out_max.get());
  // passed(result < 0.0001);
  EXPECT_EQ(result, 0);
}

void test_operators_argmax(void) {  // NT: WIP   do not use t_import int 64 here
  ctx.gc();

  // reference inputs
  ctx.addCached(hold(t_import.float_import("/fs/constants/ArgMax/in/ArgMax-input_0.idx")), "ref_a");
  ctx.addCached(hold(t_import.int_import("/fs/constants/ArgMax/in/ArgMax-dimension_0.idx")), "ref_dim");

  // reference outputs
  /// NT: FIXME: argmax outputs int64 tensor which isn't supported by
  /// int_import.
  S_TENSOR ref_out = ctx.addCached(hold(t_import.float_import("/fs/constants/ArgMax/out/ArgMax_0.idx")), "ref_out");

  // Implementation goes here

  // modify the checks below:
  S_TENSOR out = ctx.addCached(hold(new RamTensor<int>(ref_out->getShape())), "out");

  TNameList inputs = {"ref_a", "ref_dim"};
  TNameList outputs = {"out"};

  
  ctx.push_static(hold(new ArgMaxOp<float, int>()), "ArgMaxOp", inputs, outputs);
  ctx.eval();
  

  Tensor* out_float = TensorCast<int, float>(out.get());  ///NT: /WIP  how to handle the name?

  double result = meanPercentErr<float>(ref_out.get(), out_float);

  // passed(result < 0.0001);
  EXPECT_EQ(result, 0);
}

void test_operators_argmax2(void) {  // NT: WIP   do not use t_import int 64 here
  ctx.gc();

  S_TENSOR test_input = ctx.add(TensorConstant<float>({10, 5}, 0.0f), "test_input");
  *(test_input->write<float>(25, 0)) = 1.0f;
  *(test_input->write<float>(26, 0)) = 1.0f;
  *(test_input->write<float>(7, 0))  = 1.0f;
  *(test_input->write<float>(48, 0)) = 1.0f;
  *(test_input->write<float>(14, 0)) = 1.0f;

  S_TENSOR test_dim = ctx.add(new RamTensor<int>({1}), "test_dim");
  *(test_dim->write<int>(0, 0)) = 0;

  S_TENSOR test_out_ref = ctx.add(new RamTensor<float>({5}), "test_out_ref");
  *(test_out_ref->write<float>(0, 0)) = 5.0f;
  *(test_out_ref->write<float>(1, 0)) = 5.0f;
  *(test_out_ref->write<float>(2, 0)) = 1.0f;
  *(test_out_ref->write<float>(3, 0)) = 9.0f;
  *(test_out_ref->write<float>(4, 0)) = 2.0f;

  S_TENSOR test_out = ctx.add(new RamTensor<float>(test_out_ref->getShape()), "test_out");
  TNameList inputs = {"test_input", "test_dim"};
  TNameList outputs = {"test_out"};

  
  ctx.push(new ArgMaxOp<float, float>(), inputs, outputs);
  ctx.eval();
  

  double result = meanPercentErr<float>(test_out_ref.get(), test_out.get());
  EXPECT_LT(result, 0.0001);
  //EXPECT_EQ(result, 0);
}

void test_operators_add(void) {
  // reference inputs
  ctx.addCached(hold(t_import.float_import("/fs/constants/ref_add/in/Const_5_0.idx")), "a");
  ctx.addCached(hold(t_import.float_import("/fs/constants/ref_add/in/Const_6_0.idx")), "b");

  // reference outputs
  S_TENSOR ref_out = ctx.addCached(hold(t_import.float_import("/fs/constants/ref_add/out/ref_add_0.idx")), "ref_out");

  // Implementation goes here

  // modify the checks below:
  S_TENSOR out = ctx.addCached(hold(new RamTensor<float>(ref_out->getShape())), "out");
  TNameList inputs = {"a", "b"};
  TNameList outputs = {"out"};
  
  ctx.push_static(hold(new AddOp<float, float>()), "AddOp", inputs, outputs);
  ctx.eval();
  

  double result = meanPercentErr<float>(ref_out.get(), out.get());
   EXPECT_LT(result, 0.0001);
}

void test_operators_min(void) {
  

  ctx.gc();

  // reference inputs
  ctx.addCached(hold(t_import.float_import("/fs/constants/ref_min/in/Const_2_0.idx")), "a");
  ctx.addCached(hold(t_import.int_import("/fs/constants/ref_min/in/Const_3_0.idx")), "dim");

  // reference outputs
  S_TENSOR ref_out = ctx.addCached(hold(t_import.float_import("/fs/constants/ref_min/out/ref_min_0.idx")), "ref_out");


  // modify the checks below:
  S_TENSOR out = ctx.addCached(hold(new RamTensor<float>(ref_out->getShape())), "out");
  TNameList inputs = {"a", "dim"};
  TNameList outputs = {"out"};

  
  ctx.push_static(hold(new MinOp()), "MinOp", inputs, outputs);
  ctx.eval();
  

  double result = meanPercentErr<float>(ref_out.get(), out.get());
  // passed(result < 0.0001);
  EXPECT_EQ(result, 0);
}

void test_operators_max(void) {
  ctx.gc();

  // reference inputs
  ctx.addCached(hold(t_import.float_import("/fs/constants/ref_max/in/Const_2_0.idx")), "a");
  ctx.addCached(hold(t_import.int_import("/fs/constants/ref_max/in/Const_4_0.idx")), "dim");

  // reference outputs
  S_TENSOR ref_out = ctx.addCached(hold(t_import.float_import("/fs/constants/ref_max/out/ref_max_0.idx")), "ref_out");

  // Implementation goes here

  // modify the checks below:
  S_TENSOR out = ctx.addCached(hold(new RamTensor<float>(ref_out->getShape())), "out");
  TNameList inputs = {"a", "dim"};
  TNameList outputs = {"out"};
  
  ctx.push_static(hold(new MaxOp()), "MaxOp", inputs, outputs);
  ctx.eval();
  

  double result = meanPercentErr<float>(ref_out.get(), out.get());
  // passed(result < 0.0001);
  EXPECT_EQ(result, 0);
}


// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(operators, requantizationRange, "Test requantization range")
UTENSOR_TEST(operators, requantize  , "Test requantize")
UTENSOR_TEST(operators, requantize2 , "Test requantize 2")
UTENSOR_TEST(operators, argmax      , "Test argmax")
UTENSOR_TEST(operators, argmax2     , "Test argmax 2")
UTENSOR_TEST(operators, add         , "Test add")
UTENSOR_TEST(operators, min         , "Test min")
UTENSOR_TEST(operators, max         , "Test max")


// Third, run like hell
UTENSOR_TEST_RUN()
