# Cotributing to uTensor

Welcome to uTensor contribution document, there are two kinds of pull request:
  1. implement new feature on uTensor:
  *  create a new issue and discuss the idea with core maintainers. 
  *  After making consensus on,  just start the implementation of the feture.
  
  2. Extend and fix bugs on an feature : 
  *  Look and make comment on the issue which you are interested in.
  *  if you need extra help and context, please contact with core developer for help. 


Once you finish implementing a feature or bugfix, please send a Pull Request to

# Developing locally with uTensor
build locally:
1. docker pull mbartling/utensor_cli
2. locally install the uTensor, there are some steps:
    git clone https://github.com/uTensor
3. git clone https://github.com/ThrowTheSwitch/Unity.git unity_temp
4. mv unity_temp/* uTensor/unity/
5. write your own test for operator
6. find TESTS -type f -exec sed -e 's/\/fs\//TESTS\//g' -i {} \;
7. find uTensor/core/ -type f -exec sed -e 's/\/fs\//\//g' -i {} \;
8. cmake .
9. make
10. ctest -VV
    
  
  
# Unit Test
 1. write feature test case to test the execution.
 2. if the local test case is find, try to make pull-reqeust and bot will make ci for you. 



