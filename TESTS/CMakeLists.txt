add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

macro(package_add_test_with_libraries TESTNAME FILES LIBRARIES directory TEST_WORKING_DIRECTORY)
    add_executable(${TESTNAME} ${FILES})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main ${LIBRARIES})
    gtest_discover_tests(${TESTNAME}
      #WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
      #PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    target_include_directories(${TESTNAME} PUBLIC ${directory})
endmacro()

# Core
package_add_test_with_libraries(test_types core/test_types.cpp utensor_core core "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_alloc allocators/test_arenaAllocator.cpp utensor allocators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_context core/test_context.cpp utensor core "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_quantparams core/test_quantization_params.cpp utensor core "${PROJECT_DIR}/test-data/")

# Tensors
package_add_test_with_libraries(test_ramtensor tensors/test_ramtensor.cpp utensor tensors "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_buffertensor tensors/test_buffertensor.cpp utensor tensors "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_romtensor tensors/test_romtensor.cpp utensor tensors "${PROJECT_DIR}/test-data/")

# Operators
package_add_test_with_libraries(test_arith operators/test_arithmetic.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_matrix_ops operators/test_matrix.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_convolution_no_bias operators/test_convolution_nobias.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_convolution_bias operators/test_convolution_bias.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_dws_convolution operators/test_dws_convolution.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_relu operators/test_relu.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_maxpool operators/test_maxpool.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_avgpool operators/test_avgpool.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_squeeze operators/test_squeeze.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_argmax operators/test_argmax.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_argmin operators/test_argmin.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_reshape operators/test_reshape.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_dequantize operators/test_dequantize.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_quantize operators/test_quantize.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_quant_dws_conv operators/test_quantized_dws_conv.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_quant_fully_connect_2 operators/test_quant_fully_connect_2.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_fully_connect operators/test_fully_connected.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_sq_conv2d operators/test_sq_conv2d.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_sq_softmax operators/test_sq_softmax.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_sq_logistic operators/test_sq_logistic.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_sq_tanh operators/test_sq_tanh.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_transpose operators/test_transpose.cpp utensor operators "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_sum operators/test_sum.cpp utensor operators "${PROJECT_DIR}/test-data/")

# Includes
package_add_test_with_libraries(test_top_include library/test_top_include.cpp utensor library "${PROJECT_DIR}/test-data/")

# Models
package_add_test_with_libraries(test_model1 model/test_model1.cpp utensor model "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_model2 model/test_model2.cpp utensor model "${PROJECT_DIR}/test-data/")
# TODO Figure out why these tests have accuracy issues. Might just be a copy paste issue
# package_add_test_with_libraries(test_layer_1_quantize model/test_layer_1_quantize utensor model "${PROJECT_DIR}/test-data/")
#package_add_test_with_libraries(test_layer_2_dws_conv2d model/test_layer_2_dws_conv2d utensor model "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_layer_3_dws_conv2d model/test_layer_3_dws_conv2d.cpp utensor model "${PROJECT_DIR}/test-data/")
package_add_test_with_libraries(test_layer_4_maxpool model/test_layer_4_maxpool.cpp utensor model "${PROJECT_DIR}/test-data/")
# package_add_test_with_libraries(test_layer_5_qFC model/test_layer_5_qFC utensor model "${PROJECT_DIR}/test-data/")
# using integration test instead of layer-by-layer tests above
package_add_test_with_libraries(integration_test model/integration_test.cpp utensor model "${PROJECT_DIR}/test-data/")

#Error Handlers
package_add_test_with_libraries(test_simple_error_handler error_handlers/test_errorhandler.cpp utensor error_handlers "${PROJECT_DIR}/test-data/")

