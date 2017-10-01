# uTensor

## Introduction

  uTensor is an extreme light-weight Deep-Learning Inference framework built on mbed and Tensorflow.

  This project is under going constant development.

## Finding your target name

`mbed detect` to see which target is connect to the board

`mbedls -l` to list all supported targets

## Configure

See mbed_app.json

## Build Steps

1. Clone the repository
2. In the project folder, run `mbed new .`
3. Run `mbed deploy` to download all referenced libraries
4. Insert the prepared SD card to the board (see SD Card Preparation Section)
4. Use `mbed compile -t GCC_ARM -m NUCLEO_F767ZI --profile=./build_profile/release.json` to build for K64F. Or, `mbed compile -t GCC_ARM -m NUCLEO_F767ZI --profile=./build_profile/release.json -f` to compile and flash

## SD Card Preparation
The test data has to be loaded to the SD card for the default binary to run:

1. Go to the `project]\TESTS\scripts` folder
2. Run `python3 compileTestData.py`, the directory `[project]\TESTS\scripts\testData` will be created.
3. Copy `[project]\TESTS\scripts\testData` to the root of your SD card.

You should have an internal or external SD card reader for the board. The current SD driver only support SD card upto 32GB.
