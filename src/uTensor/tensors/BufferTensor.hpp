#ifndef UTENSOR_BUFFER_TENSOR_H
#define UTENSOR_BUFFER_TENSOR_H

#include "tensorBase.hpp"

namespace uTensor { 
class BufferTensor : public TensorInterface {
    protected:
        virtual void* read(uint32_t linear_index) const override;
        virtual void* write(uint32_t linear_index) override;

    public:
        BufferTensor(TensorShape _shape, ttype _type);
        BufferTensor(TensorShape _shape, ttype _type, void* buffer);

        bool is_bound_to_buffer() const;
        // Check if is bound to this pointer
        bool is_bound_to_buffer(void* b) const;
        bool bind(void* b);
        bool unbind();

        virtual ~BufferTensor();
        // Does nothing
        virtual void resize(TensorShape new_shape) override;
    private:
        uint8_t* _buffer;
};
}
#endif
