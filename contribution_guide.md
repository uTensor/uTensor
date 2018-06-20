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

1. locally install the uTensor, there are some steps:
    `git clone https://github.com/uTensor`
2. install circleci tool locally, please refer [Circleci](https://circleci.com/docs/2.0/local-cli/#installing-the-circleci-local-cli-on-macos-and-linux-distros)
2. In uTensor directory, run `circleci build --job build`  

Please refer to test guideline (https://github.com/uTensor/uTensor/tree/develop/TESTS/README.md) for more information


