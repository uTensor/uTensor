#include "NnOps.hpp"

void Softmax(S_TENSOR input, S_TENSOR output)
{
    if(input->getDim() != 1)
    {
        for(int i = 0; i < input->getDim() - 1; i++)
        {
            if(input->getShape().at(i) != 1)
            {
                ERR_EXIT("Softmax is supported only for flatten Tensor");
            }
        }
    }

    if (output && output->getSize() == 0) {
        output->resize(input->getShape());
    }


    float* out_ptr = output->write<float>(0,0);
    const float* in_ptr = input->read<float>(0, 0);

    float reduced_sum = 0;
    for(int i = 0; i < input->getSize(); i++)
    {
        reduced_sum += exp(in_ptr[i]);
    }

    for(int i = 0; i < output->getSize(); i++)
    {
        out_ptr[i] = exp(in_ptr[i]) / reduced_sum;
    }

}
