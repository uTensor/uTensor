#ifndef UTENSOR_ROM_TENSOR_H
#define UTENSOR_ROM_TENSOR_H
#include "BufferTensor.hpp"

namespace uTensor {

// Literally same behavior as buffer tensor except writing is prohibited
class RomTensor : public BufferTensor {
    protected:
        virtual void* write(uint32_t linear_index) override;
    public:
        RomTensor(TensorShape _shape, ttype _type, const void* buffer);
        virtual ~RomTensor();
        virtual void resize(TensorShape new_shape) override;
};
}
#endif
