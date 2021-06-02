#ifndef UTENSOR_RESHAPE_H
#define UTENSOR_RESHAPE_H

#include "context.hpp"
#include "operatorBase.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "uTensor_util.hpp"

using std::array;

namespace uTensor {
/*
 * Requires proper testing, tested only for some cases
 * Not supported features: Elipses_axis mask, new_axis_mask, undefined dimension
 * size (dim == -1)
 */
template <typename T>
void stridedslice_kernel(Tensor& input, Tensor& begin_tensor,
                         Tensor& end_tensor, Tensor& strides_tensor,
                         Tensor& output, int begin_mask, int ellipsis_mask,
                         int end_mask, int new_axis_mask,
                         int shrink_axis_mask) {
  TensorShape& begin_tensor_shape = begin_tensor->get_shape();
  uint32_t begin_tensor_size = begin_tensor_shape.num_elems();

  TensorShape& end_tensor_shape = end_tensor->get_shape();
  uint32_t end_tensor_size = end_tensor_shape.num_elems();

  TensorShape& strides_tensor_shape = strides_tensor->get_shape();
  uint32_t strides_tensor_size = strides_tensor_shape.num_elems();

  if (begin_tensor_shape.num_dims() != 1 || end_tensor_shape.num_dims() != 1 ||
      strides_tensor_shape.num_dims() != 1 ||
      strides_tensor_size != begin_tensor_size ||
      strides_tensor_size != end_tensor_size ||
      strides_tensor_size != input->get_shape().num_dims()) {
    ERR_EXIT(
        "StridedSlice: Expected begin, end, and strides to be 1D equal size "
        "tensors, which size is equal to input_tensor dimensionality (new mask "
        "&& elipsis mask not supported). "
        "Input dimensionality:\"%d\", begin_tensor size: \"%d\", end_tensor "
        "size: \"%d\" and strides_tensor size: \"%d\".",
        input->getDim(), begin_tensor->getSize(), end_tensor->getSize(),
        strides_tensor->getSize());
  }

  if (input->get_shape().num_dims() >= 32) {
    ERR_EXIT(
        "StridedSlice: Only 32bit masks are supported, input vector cannot "
        "have higher dimensionality than 32")
  }

  // TODO
  if (ellipsis_mask) {
    ERR_EXIT("StridedSlice: Ellipsis mask not supported yet");
  }

  // TODO
  if (new_axis_mask) {
    ERR_EXIT("StridedSlice: New axis mask not supported yet")
  }

  output->resize(input->getShape());  ///????????
  memcpy(output->write<void>(0, 0), input->read<void>(0, 0),
         input->getSize() * sizeof(T));

  const int* begin_ptr = begin_tensor->read<int>(0, 0);
  const int* end_ptr = end_tensor->read<int>(0, 0);
  const int* strides_ptr = strides_tensor->read<int>(0, 0);
  for (int i = 0; i < input->getDim(); ++i) {
    int begin_i = begin_ptr[i];
    int end_i = end_ptr[i];
    int stride_i = strides_ptr[i];
    int32_t dim_i = input->getShape().at(i);

    // if (stride_i == 0) {
    //     ERR_EXIT("StridedSlice: strides[\"%d\"] must be non-zero", i);
    // }

    // bool shrink_i = (shrink_axis_mask & (1 << i));
    // if(shrink_i)
    // {
    //     end_i = begin_i + 1;
    //     stride_i = 1;

    // }

    // bool begin_masked_i = (begin_mask & (1 << i));
    // bool end_masked_i = (end_mask & (1 << i));
    // const std::array<int64_t, 2> valid_range = {
    //     {stride_i > 0 ? 0 : -1, stride_i > 0 ? (int64_t)dim_i :
    //     (int64_t)(-dim_i - 1)}};

    if (begin_masked_i) begin_i = valid_range[0];
    if (end_masked_i) end_i = valid_range[1];

    int64_t interval_length = (end_i - begin_i);
    int64_t size_i =
        interval_length / stride_i + (interval_length % stride_i != 0 ? 1 : 0);

    if (size_i == dim_i && stride_i == 1)
      continue;  // whole dimension is taken unchanged
    else {
      std::vector<int> indexes;
      // Reversing the dimension
      if (begin_i < 0 && end_i < 0 && stride_i < 0) {
        begin_i += dim_i;
        end_i += dim_i;
        for (int j = 0; j < size_i; j++) {
          int index = begin_i + j * stride_i;
          if (index <= end_i) index = end_i + 1;
          indexes.push_back(index);
        }
      } else if (begin_i >= 0 && end_i > 0 && stride_i > 0) {
        for (int j = 0; j < size_i; j++) {
          int index = begin_i + j * stride_i;
          if (index >= end_i) index = end_i - 1;
          indexes.push_back(index);
        }
      } else {
        ERR_EXIT(
            "Slided Strice: begin, and and stride must be all less than zero "
            "for the dimension or none of them")
      }

      S_TENSOR dim_input(new RamTensor<T>(output->getShape()));
      memcpy(dim_input->write<T>(0, 0), output->read<T>(0, 0),
             output->getSize() * sizeof(T));
      StridedSlice1D<T>(dim_input, output, i, indexes);
    }
  }

  TensorShape shrinked_shape;
  for (int i = 0; i < output->getDim(); i++) {
    bool shrink_i = (shrink_axis_mask & (1 << i));
    if (!shrink_i) {
      shrinked_shape.push_back(output->getShape().at(i));
    }
  }
  if (shrinked_shape.size() == 0) shrinked_shape.push_back(1);
  // the shrinked dimension size was 1 so no data update needed
  output->resize(shrinked_shape);
}
}  // namespace uTensor
}

#endif  // UTENSOR_RESHAPE_H
