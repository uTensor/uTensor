
import os
import pickle
from collections import namedtuple
import numpy as np

TFLM_Tensor = namedtuple('TFLM_Tensor', ['tensor', 'quantization'])

def import_test_data(dir_path):
    test = {}
    with open(dir_path + '/inputs.pkl','rb') as pickle_in:
        test["inputs"] = pickle.load(pickle_in)
    with open(dir_path + '/outputs.pkl','rb') as pickle_in:
        test["outputs"] = pickle.load(pickle_in)
    with open(dir_path + '/option.pkl','rb') as pickle_in:
        test["option"] = pickle.load(pickle_in)
    return test

if __name__ == '__main__':
    test = import_test_data("1_DEPTHWISE_CONV_2D")
    #all input tensor names, in order
    print("input tensor names: ", test['inputs'].keys())
    #the last input tensor as numpy array
    tensor_info = test['inputs']['StatefulPartitionedCall/my_model/conv2d/Conv2D_bias']
    print("tensor: ", tensor_info.tensor)
    print("shape: ", tensor_info.tensor.shape)
    print("dtype: ", tensor_info.tensor.dtype)
    #quantization info
    print("quantization: ", tensor_info.quantization)
    #op configuration
    print("options:")
    print(test["option"])