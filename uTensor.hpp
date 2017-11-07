#ifndef UTENSOR_HPP
#define UTENSOR_HPP
#include <random>
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "mbed.h"
#include "stdio.h"
#include "deep_mnist_mlp.hpp"


class uTensor{
    public:
        uTensor();
        ~uTensor();
    private:
        Serial pc;
        SDBlockDevice bd;
        FATFileSystem fs;

};

#endif
