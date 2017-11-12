#ifndef UTENSOR_MATH_TESTS
#define UTENSOR_MATH_TESTS

#include "MathOps.hpp"
#include "tensorIdxImporter.hpp"
#include "test.hpp"
#include "context.hpp"

class MathOpsTest : public Test {
    TensorIdxImporter t_import;
    Context ctx;
 public:
  void requantization_rangeTest(void) {
    testStart("requantization_range");

    // reference inputs
    TENSOR a =
        ctx.add(t_import.int_import("/fs/testData/rqRange/in/qMatMul_0.idx"));
    TENSOR a_min =
        ctx.add(t_import.float_import("/fs/testData/rqRange/in/qMatMul_1.idx"));
    TENSOR a_max =
        ctx.add(t_import.float_import("/fs/testData/rqRange/in/qMatMul_2.idx"));

    // reference outputs
    TENSOR ref_min =
        ctx.add(t_import.float_import("/fs/testData/rqRange/out/rqRange_0.idx"));
    TENSOR ref_max =
        ctx.add(t_import.float_import("/fs/testData/rqRange/out/rqRange_1.idx"));

    // Implementation goes here

    // modify the checks below:
    TENSOR out_min = ctx.add(new RamTensor<float>(ref_min.lock()->getShape()));
    TENSOR out_max = ctx.add(new RamTensor<float>(ref_max.lock()->getShape()));
    TList inputs = {a, a_min, a_max};
    TList outputs = {out_min, out_max};

    S_TENSOR ref_min_val = ref_min.lock();
    S_TENSOR ref_max_val = ref_max.lock();
    S_TENSOR out_min_val = out_min.lock();
    S_TENSOR out_max_val = out_max.lock();
    
    timer_start();
    ctx.push(new Requantization_RangeOp(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result =
        meanPercentErr<float>(ref_min_val.get(), out_min_val.get()) + meanPercentErr<float>(ref_max_val.get(), out_max_val.get());
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void requantizeTest(void) {
    testStart("requantize");

    // reference inputs
    TENSOR a = ctx.add(t_import.int_import("/fs/testData/rQ/in/qMatMul_0.idx"));
    TENSOR a_min =
        ctx.add(t_import.float_import("/fs/testData/rQ/in/qMatMul_1.idx"));
    TENSOR a_max =
        ctx.add(t_import.float_import("/fs/testData/rQ/in/qMatMul_2.idx"));
    TENSOR r_a_min =
       ctx.add(t_import.float_import("/fs/testData/rQ/in/rqRange_0.idx"));
    TENSOR r_a_max =
        ctx.add(t_import.float_import("/fs/testData/rQ/in/rqRange_1.idx"));
    // tf.quint8

    // reference outputs
    TENSOR ref_a_q =
        ctx.add(t_import.ubyte_import("/fs/testData/rQ/out/rQ_0.idx"));
    TENSOR ref_a_min =
        ctx.add(t_import.float_import("/fs/testData/rQ/out/rQ_1.idx"));
    TENSOR ref_a_max =
        ctx.add(t_import.float_import("/fs/testData/rQ/out/rQ_2.idx"));

    // modify the checks below:
    TENSOR a_q = ctx.add(new RamTensor<unsigned char>(ref_a_q.lock()->getShape()));
    TENSOR a_min_q = ctx.add(new RamTensor<float>(ref_a_min.lock()->getShape()));
    TENSOR a_max_q = ctx.add(new RamTensor<float>(ref_a_max.lock()->getShape()));

    TList inputs = {a, a_min, a_max, r_a_min, r_a_max};
    TList outputs = {a_q, a_min_q, a_max_q};

    S_TENSOR ref_a = ref_a_q.lock();
    S_TENSOR out_a = a_q.lock();
    S_TENSOR ref_min = ref_a_min.lock();
    S_TENSOR out_min = a_min_q.lock();
    S_TENSOR ref_max = ref_a_max.lock();
    S_TENSOR out_max = a_max_q.lock();
    // Implementation goes here
    timer_start();
    ctx.push(new RequantizeOp(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<unsigned char>(ref_a.get(), out_a.get()) +
                    meanPercentErr<float>(ref_min.get(), out_min.get()) +
                    meanPercentErr<float>(ref_max.get(), out_max.get());
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void requantizeTest2(void) {
    testStart("requantize2");

    // reference inputs
    TENSOR  a = ctx.add(t_import.int_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_0.idx"));
    TENSOR a_min =
        ctx.add(t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_1.idx"));
    TENSOR a_max =
        ctx.add(t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_2.idx"));
    TENSOR r_a_min =
        ctx.add(t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_0.idx"));
    TENSOR r_a_max =
        ctx.add(t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_1.idx"));
    // tf.quint8

    // reference outputs
    TENSOR ref_a_q =
        ctx.add(t_import.ubyte_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_0.idx"));
    TENSOR ref_a_min =
        ctx.add(t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_1.idx"));
    TENSOR ref_a_max =
        ctx.add(t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_2.idx"));

    
    // modify the checks below:
    TENSOR a_q = ctx.add(new RamTensor<unsigned char>(ref_a_q.lock()->getShape()));
    TENSOR a_min_q = ctx.add(new RamTensor<float>(ref_a_min.lock()->getShape()));
    TENSOR a_max_q = ctx.add(new RamTensor<float>(ref_a_max.lock()->getShape()));
    TList inputs = {a, a_min, a_max, r_a_min, r_a_max};
    TList outputs = {a_q, a_min_q, a_max_q};
    S_TENSOR ref_val = ref_a_q.lock();
    S_TENSOR ref_min = ref_a_min.lock();
    S_TENSOR ref_max = ref_a_max.lock();
    S_TENSOR out_val = a_q.lock();
    S_TENSOR out_min = a_min_q.lock();
    S_TENSOR out_max = a_max_q.lock();

    // Implementation goes here
    timer_start();
    ctx.push(new RequantizeOp(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result;
    if((result = meanPercentErr<unsigned char>(ref_val.get(), out_val.get())) != 0) {
        printf("Requantize a_q failed (%.6f)\r\n", result);
        unsigned char* ref_ptr = ref_val.get()->write<unsigned char>(0, 0);
        unsigned char* test_ptr = out_val.get()->write<unsigned char>(0, 0);
        for(uint32_t i = 0; i < ref_val->getSize(); i++) {
            if(ref_ptr[i] != test_ptr[i]) {
                printf("%lu: %d != %d\r\n", i, ref_ptr[i], test_ptr[i]);
            } else {
                printf("%lu: %d == %d\r\n", i, ref_ptr[i], test_ptr[i]);
            }
        }
    }


    if((result = meanPercentErr<float>(ref_min.get(), out_min.get())) != 0) printf("Requantize a_min_q failed (%.6f)\r\n", result);

    if((result = meanPercentErr<float>(ref_max.get(), out_max.get())) != 0) printf("Requantize a_max_q failed (%.6f)\r\n", result);

    result = meanPercentErr<unsigned char>(ref_val.get(), out_val.get()) +
                    meanPercentErr<float>(ref_min.get(), out_min.get()) +
                    meanPercentErr<float>(ref_max.get(), out_max.get());
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void argmaxTest(void) {  // NT: WIP   do not use t_import int 64 here
    testStart("argmax");

    // reference inputs
    TENSOR ref_a = ctx.add(t_import.float_import("/fs/testData/ArgMax/in/ArgMax-input_0.idx"));
    TENSOR ref_dim = ctx.add(t_import.int_import("/fs/testData/ArgMax/in/ArgMax-dimension_0.idx"));

    // reference outputs
    /// NT: FIXME: argmax outputs int64 tensor which isn't supported by
    /// int_import.
    TENSOR ref_out = ctx.add(t_import.float_import("/fs/testData/ArgMax/out/ArgMax_0.idx"));

    // Implementation goes here

    // modify the checks below:
    TENSOR out = ctx.add(new RamTensor<int>(ref_out.lock()->getShape()));
    
    TList inputs = {ref_a, ref_dim};
    TList outputs = {out};

    S_TENSOR ref_val = ref_out.lock();
    S_TENSOR out_val = out.lock();
    timer_start();
    ctx.push(new ArgMaxOp<float, int>(), inputs, outputs);
    ctx.eval();
    timer_stop();
    
    Tensor* out_float = TensorCast<int, float>(out_val.get());

    double result = meanPercentErr<float>(ref_val.get(), out_float);

    // passed(result < 0.0001);
    passed(result == 0);
  }

  void argmaxTest2(void) {  // NT: WIP   do not use t_import int 64 here
    testStart("argmax2");
    TENSOR test_input = ctx.add(TensorConstant<float>({10, 5}, 0.0f));
    *(test_input.lock()->write<float>(25, 0)) = 1.0f;
    *(test_input.lock()->write<float>(26, 0)) = 1.0f;
    *(test_input.lock()->write<float>(7, 0)) = 1.0f;
    *(test_input.lock()->write<float>(48, 0)) = 1.0f;
    *(test_input.lock()->write<float>(14, 0)) = 1.0f;

    TENSOR test_dim = ctx.add(new RamTensor<int>({1}));
    *(test_dim.lock()->write<int>(0, 0)) = 0;

    TENSOR test_out_ref = ctx.add(new RamTensor<float>({5}));
    *(test_out_ref.lock()->write<float>(0, 0)) = 5.0f;
    *(test_out_ref.lock()->write<float>(1, 0)) = 5.0f;
    *(test_out_ref.lock()->write<float>(2, 0)) = 1.0f;
    *(test_out_ref.lock()->write<float>(3, 0)) = 9.0f;
    *(test_out_ref.lock()->write<float>(4, 0)) = 2.0f;

    TENSOR test_out = ctx.add(new RamTensor<float>(test_out_ref.lock()->getShape()));
    TList inputs = {test_input, test_dim};
    TList outputs = {test_out};
    S_TENSOR ref_val = test_out_ref.lock();
    S_TENSOR out_val = test_out.lock();

    timer_start();
    ctx.push(new ArgMaxOp<float, float>(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<float>(ref_val.get(), out_val.get());
    std::cout << result << std::endl;
     passed(result < 0.0001);
    //passed(result == 0);
  }

  void addTest(void) {
    testStart("add");

    // reference inputs
    TENSOR a =
        ctx.add(t_import.float_import("/fs/testData/ref_add/in/Const_5_0.idx"));
    TENSOR b =
        ctx.add(t_import.float_import("/fs/testData/ref_add/in/Const_6_0.idx"));

    // reference outputs
    TENSOR ref_out =
        ctx.add(t_import.float_import("/fs/testData/ref_add/out/ref_add_0.idx"));

    // Implementation goes here

    // modify the checks below:
    TENSOR out = ctx.add(new RamTensor<float>(ref_out.lock()->getShape()));
    S_TENSOR out_vxx = out.lock();
    S_TENSOR ref_vxx = ref_out.lock();
    TList inputs = {a, b};
    TList outputs = {out};
    timer_start();
    ctx.push(new AddOp<float, float>(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<float>(ref_vxx.get(), out_vxx.get());
    std::cout << result << std::endl;
     passed(result < 0.0001);
    //passed(result == 0);
  }

  void minTest(void) {
    testStart("min");

    // reference inputs
    TENSOR a =
        ctx.add(t_import.float_import("/fs/testData/ref_min/in/Const_2_0.idx"));
    TENSOR dim =
        ctx.add(t_import.int_import("/fs/testData/ref_min/in/Const_3_0.idx"));

    // reference outputs
    TENSOR ref_out =
        ctx.add(t_import.float_import("/fs/testData/ref_min/out/ref_min_0.idx"));

    // Implementation goes here

    // modify the checks below:
    TENSOR out = ctx.add(new RamTensor<float>(ref_out.lock()->getShape()));
    TList inputs = {a, dim};
    TList outputs = {out};
    S_TENSOR ref_val = ref_out.lock();
    S_TENSOR out_val = out.lock();
    timer_start();
    ctx.push(new MinOp(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<float>(ref_val.get(), out_val.get());
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void maxTest(void) {
    testStart("max");

    // reference inputs
    TENSOR a =
        ctx.add(t_import.float_import("/fs/testData/ref_max/in/Const_2_0.idx"));
    TENSOR dim =
        ctx.add(t_import.int_import("/fs/testData/ref_max/in/Const_4_0.idx"));

    // reference outputs
    TENSOR ref_out =
        ctx.add(t_import.float_import("/fs/testData/ref_max/out/ref_max_0.idx"));

    // Implementation goes here

    // modify the checks below:
    TENSOR out = ctx.add(new RamTensor<float>(ref_out.lock()->getShape()));
    TList inputs = {a, dim};
    TList outputs = {out};
    S_TENSOR ref_val = ref_out.lock();
    S_TENSOR out_val = out.lock();
    timer_start();
    ctx.push(new MaxOp(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<float>(ref_val.get(), out_val.get());
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void runAll(void) {
    argmaxTest();
    argmaxTest2();
    requantization_rangeTest();
    requantizeTest();
    requantizeTest2();
    addTest();
    minTest();
    maxTest();
  }
};

#endif  // UTENSOR_MATH_TESTS
