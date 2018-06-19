# Testing uTensor

## Introduction
Tests in uTensor are meant to blend the lines between embedded and traditional platform testing. To accomplish this we introduce a soft wrapper around Googles GTest framework and mbed Greentea.

The test runner locates tests using the following expression:
`TEST_SOURCES=TESTS/*/*/test_*.cpp`, likewise you may provide cli generated models in a `TESTS/*/*/models` directory. We like to think of this directory structure as `TESTS/TEST_GROUP/TEST_CASE/test_TEST_CASE.cpp`. 

Please refer to `TESTS/dummy/sanity/test_dummy_sanity.cpp` for a basic skeleton test.
