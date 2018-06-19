# Contributing to uTensor

Welcome to uTensor contribution guide. Want to make things better? [Talk to us](utensor@googlegroups.com)!

## Ways to Contribute

### Two kinds of pull request

  1. Adding new features to uTensor:
    -  Create a new [issue](https://github.com/uTensor/uTensor/issues) and discuss the idea with core maintainers. We want to make sure your time is well spent and learn about you.
    -  Start the implementing the feature once the feedback is received.
      - uTensor runtime [local build instructions](#localBuild)
      - Code Generator Developer Build
    -  You may need to supply your test cases if there isn't already one.
      - uTensor Runtime [test cases](https://github.com/uTensor/uTensor/tree/develop/TESTS)
      - Code Generator [test cases](https://github.com/uTensor/utensor_cgen/tree/develop/tests)
  
  2. Bug fixes and Feature extensions: 
    -  Search for the issue that you are interested in.
    -  Comment on the issue. The core developers are happy to discuss and help you when you are stuck.

We do things via [PRs](https://help.github.com/articles/about-pull-requests/). Please open a PR to the repository when you are ready for the next step.

If you would like to create a new uTensor example repository, please initiate this via an [issue discussion](https://github.com/uTensor/uTensor/issues) in the uTensor repository.


# <a name="localBuild"></a> Developing locally with uTensor

## For uTensor Runtime Library

1. `docker pull mbartling/utensor_cli`
2. locally install the uTensor, there are some steps:
    `git clone https://github.com/uTensor`
3. `git clone https://github.com/ThrowTheSwitch/Unity.git unity_temp`
4. `mv unity_temp/* uTensor/unity/`
5. `write your own test for operator`
6. `find TESTS -type f -exec sed -e 's/\/fs\//TESTS\//g' -i {} \;`
7. `find uTensor/core/ -type f -exec sed -e 's/\/fs\//\//g' -i {} \;`
8. `cmake .`
9. `make`
10. `ctest -VV`
  
Please refer to test guideline (https://github.com/uTensor/uTensor/tree/develop/TESTS/README.md) for more information


