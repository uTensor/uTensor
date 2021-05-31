#include <cstdio>
#include <iostream>
#include <numeric>
#include <typeinfo>

#include "BufferTensor.hpp"
#include "Matrix_kernels.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "operatorBase.hpp"

using namespace std;
using namespace uTensor;

// MyAddOperator is an operator which takes 2 input tensors and produce 1 output
// tensor. Normally, for reusability, we will implement a kernel function and
// invoke the function in the compute method
void my_add_kernel(const Tensor &tensor1, const Tensor &tensor2,
                   Tensor &output) {
  printf("my_add_kernel invoked!\n");
  TensorShape shape1 = tensor1->get_shape();
  TensorShape shape2 = tensor2->get_shape();
  // input shapes should be the same
  if (shape1 != shape2) {
    // you can throw error as following:
    Context::get_default_context()->throwError(new InvalidReshapeError);
    // Please refer to error_handling tutorial to see how to set a error handler
    // to handle the error in uTensor. Without an handler, all errors are
    // ignored by default.
    return;
  }
  if (tensor1->get_type() != flt || tensor2->get_type() != flt ||
      output->get_type() != flt) {
    // we only support float tensors for MyAddOperator
    return;
  }
  for (uint32_t i = 0; i < tensor1->num_elems(); ++i) {
    // When reading from tensors, you must cast it to the expected data types.
    // You can access tensor element by flatten index:
    float a = static_cast<float>(tensor1(i));
    // or by indices of axis:
    float b = static_cast<float>(tensor2(0, i));
    // write result to output tensor
    output(i) = a + b;
  }
}

template <typename T>
void my_matmal_kernel(const Tensor &a, const Tensor &b, Tensor &c) {
  // Decide on c shape
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  TensorShape c_shape = c->get_shape();
  if (a_shape.num_dims() > 2 || b_shape.num_dims() > 2 ||
      c_shape.num_dims() > 2 || a_shape[1] != b_shape[0] ||
      a_shape[0] != c_shape[0] || b_shape[1] != c_shape[1]) {
    uTensor_printf("[Error] Invalid matrix multiple shape mismatch\n");
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }

  for (uint32_t i = 0; i < a_shape[0]; i++) {
    for (uint32_t j = 0; j < b_shape[1]; j++) {
      // c(i, j) = static_cast<T>(0);
      T tmp = 0;
      for (uint32_t k = 0; k < a_shape[1]; k++) {
        // cout << "a(i, k):" << a(i, k) << endl;

        cout << "static_cast<T>(a(i, k)):" << static_cast<T>(a(i, k)) << endl;
        // cout << "b(k, j):" << b(k, j) << endl;

        cout << "static_cast<T>(b(k, j)):" << static_cast<T>(b(k, j)) << endl;
        cout << "static_cast<T>(b(0, j * k)):" << static_cast<T>(b(0, j * k))
             << endl;
        tmp += static_cast<T>(a(i, k)) * static_cast<T>(b(k, j));

        // printf("i, j, k : %d %d %d %d %d\n", i, j, k, static_cast<T>(a(i, k))
        // , static_cast<T>(b(k, j)));
      }
      // c(i*j) = tmp;
      c(i, j) = tmp;
      cout << "tmp to assignment: " << tmp << endl;
      tmp = c(i, j);
      cout << "c(i, j) from c: " << tmp << endl;
    }
  }
}

template <typename T>
void my_concat_kernel(const Tensor &a, const Tensor &b, const Tensor &axis,
                      Tensor &c) {
  // Decide on c shape
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  TensorShape c_shape = c->get_shape();
  if (a_shape.num_dims() > 2 || b_shape.num_dims() > 2 ||
      c_shape.num_dims() > 2 || axis > a_shape.num_dims() ||
      axis > b_shape.num_dims() || axis > c_shape.num_dims() ||
      (a_shape[axis] + b_shape[axis]) == c_shape[axis]) {
    uTensor_printf("[Error] Invalid matrix concat shape mismatch\n");
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }
  cout << "Start my_concat_kernel" << endl;

  for (uint32_t i = 0; i < a_shape.num_dims(); ++i) {
    cout << "a_shape[" << i << "]: " << a_shape[i] << endl;
  }

  for (uint32_t i = 0; i < b_shape.num_dims(); ++i) {
    cout << "b_shape[" << i << "]: " << b_shape[i] << endl;
  }

  for (uint32_t i = 0; i < c_shape.num_dims(); ++i) {
    cout << "c_shape[" << i << "]: " << c_shape[i] << endl;
  }

  // /*
  // python stype psudo code
  // */

  // if axis == 0:
  //   for i in range():
  //     if i < a.shape[0]:
  //       for j in range():
  //         for k in range():
  //           tmp = a[i,j,k]
  //     else:
  //       new_i = i-a.shape[0]
  //       for j in range():
  //         for k in range():
  //           tmp = b[new_i,j,k]
  // elif axis == 1:
  //   for j in range():
  //     if j < a.shape[1]:
  //       for i in range():
  //         for k in range():
  //           tmp = a[i,j,k]
  //     else:
  //       new_j = j-a.shape[1]
  //       for i in range():
  //         for k in range():
  //           tmp = b[i,new_j,k]
  // elif axis == 2:
  //   for k in range():
  //     if k < a.shape[2]:
  //       for i in range():
  //         for j in range():
  //           tmp = a[i,j,k]
  //     else:
  //       new_k = k-a.shape[2]
  //       for i in range():
  //         for j in range():
  //           tmp = b[i,j,new_k]
  // else:
  //   print("axis is invalid!!")

  /*
  C++ implementation
  */
  cout << "c_shape.num_dims():" << static_cast<char>(c_shape.num_dims())
       << "????" << endl;
  uint32_t int_axis = axis(0);
  switch (c_shape.num_dims()) {
    /*
    for 3 dimension metrix.
    */
    case 3:
      switch (int_axis) {
        case 0:
          cout << "case 0:" << endl;
          for (uint32_t i = 0; i < c_shape[0]; i++) {
            if (i < a_shape[0]) {
              for (uint32_t j = 0; j < c_shape[1]; j++) {
                for (uint32_t k = 0; k < c_shape[2]; k++) {
                  // cout << "a("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(a(i,j,k)) << endl;
                  c(i, j, k) = static_cast<T>(a(i, j, k));
                  // cout << "c("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(c(i,j,k)) << endl;
                }
              }
            } else {
              int new_i = i - a_shape[0];
              for (uint32_t j = 0; j < c_shape[1]; j++) {
                for (uint32_t k = 0; k < c_shape[2]; k++) {
                  // cout << "b("<< new_i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(b(new_i,j,k)) << endl;
                  c(i, j, k) = static_cast<T>(b(new_i, j, k));
                  // cout << "c("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(c(i,j,k)) << endl;
                }
              }
            }
          }
          break;
        case 1:
          for (uint32_t j = 0; j < c_shape[1]; j++) {
            if (j < a_shape[1]) {
              for (uint32_t i = 0; i < c_shape[0]; i++) {
                for (uint32_t k = 0; k < c_shape[2]; k++) {
                  // cout << "a("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(a(i,j,k)) << endl;
                  c(i, j, k) = static_cast<T>(a(i, j, k));
                  // cout << "c("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(c(i,j,k)) << endl;
                }
              }
            } else {
              int new_j = j - a_shape[1];
              for (uint32_t i = 0; i < c_shape[0]; i++) {
                for (uint32_t k = 0; k < c_shape[2]; k++) {
                  // cout << "b("<< i << ", " << new_j << ", "<< k << ") = " <<
                  // static_cast<T>(b(i,new_j,k)) << endl;
                  c(i, j, k) = static_cast<T>(b(i, new_j, k));
                  // cout << "c("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(c(i,j,k)) << endl;
                }
              }
            }
          }
          break;
        case 2:
          for (uint32_t k = 0; k < c_shape[2]; k++) {
            if (k < a_shape[2]) {
              for (uint32_t i = 0; i < c_shape[0]; i++) {
                for (uint32_t j = 0; j < c_shape[1]; j++) {
                  // cout << "a("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(a(i,j,k)) << endl;
                  c(i, j, k) = static_cast<T>(a(i, j, k));
                  // cout << "c("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(c(i,j,k)) << endl;
                }
              }
            } else {
              int new_k = k - a_shape[2];
              for (uint32_t i = 0; i < c_shape[0]; i++) {
                for (uint32_t j = 0; j < c_shape[1]; j++) {
                  // cout << "b("<< i << ", " << j << ", "<< new_k << ") = " <<
                  // static_cast<T>(a(i,j,new_k)) << endl;
                  c(i, j, k) = static_cast<T>(b(i, j, new_k));
                  // cout << "c("<< i << ", " << j << ", "<< k << ") = " <<
                  // static_cast<T>(c(i,j,k)) << endl;
                }
              }
            }
          }
          break;
      }
      break;
    /*
    for 1 dimension metrix.
    */
    case 2:
      switch (int_axis) {
        case 0:
          cout << "case 2:0!!:" << endl;
          for (uint32_t i = 0; i < c_shape[0]; i++) {
            if (i < a_shape[0]) {
              for (uint32_t j = 0; j < c_shape[1]; j++) {
                // cout << "a("<< i << ", " << j << "):" <<
                // static_cast<T>(a(i,j)) << endl;
                c(i, j) = static_cast<T>(a(i, j));
                // cout << "c("<< i << ", " << j << ") = " <<
                // static_cast<T>(c(i,j)) << endl;
              }
            }

            else {
              int new_i = i - a_shape[0];
              for (uint32_t j = 0; j < c_shape[1]; j++) {
                // cout << "a("<< new_i << ", " << j << "):" <<
                // static_cast<T>(b(new_i,j)) << endl;
                c(i, j) = static_cast<T>(b(new_i, j));
                // cout << "c("<< i << ", " << j << ") = " <<
                // static_cast<T>(c(i,j)) << endl;
              }
            }
          }
          break;
        case 1:
          cout << "case 2:1!!:" << endl;
          for (uint32_t j = 0; j < c_shape[1]; j++) {
            if (j < a_shape[1]) {
              // cout << "take from a!!:" << endl;
              for (uint32_t i = 0; i < c_shape[0]; i++) {
                // cout << "a("<< i << ", " << j << "):" <<
                // static_cast<T>(a(i,j)) << endl;
                c(i, j) = static_cast<T>(a(i, j));
                // cout << "c("<< i << ", " << j << ") = " <<
                // static_cast<T>(c(i,j)) << endl;
              }
            }

            else {
              // cout << "take from b!!:" << endl;
              int new_j = j - a_shape[1];
              for (uint32_t i = 0; i < c_shape[0]; i++) {
                // cout << "b("<< i << ", " << new_j << "):" <<
                // static_cast<T>(b(i,new_j)) << endl;
                c(i, j) = static_cast<T>(b(i, new_j));
                // cout << "c("<< i << ", " << j << ") = " <<
                // static_cast<T>(c(i,j)) << endl;
              }
            }
          }
          break;
      }
      break;
    /*
    for 1 dimension metrix.
    */
    case 1:
      for (uint32_t i = 0; i < c_shape[0]; i++) {
        if (i < a_shape[0]) {
          // cout << "a("<< i << "):" << static_cast<T>(a(i)) << endl;
          c(i) = static_cast<T>(a(i));
          // cout << "c("<< i << ") = " << static_cast<T>(c(i)) << endl;
        }

        else {
          int new_i = i - a_shape[0];
          // cout << "b("<< i <<  "):" << static_cast<T>(b(new_i)) << endl;
          c(i) = static_cast<T>(b(new_i));
          // cout << "c("<< i <<  ") = " << static_cast<T>(c(i)) << endl;
        }
      }
      break;
  }
}

// class MyAddOperator : public OperatorInterface<2, 1> {
//  public:
//   // identifiers for setting up the input tensors
//   enum names_in { a, b };
//   // identifiers for setting up the output tensors
//   enum names_out { out };

//  protected:
//   void compute() {
//     // you can retrieve input/output tensors by its identifier

//     my_add_kernel(inputs[a].tensor(), inputs[b].tensor(),
//                   outputs[out].tensor());

//     printf("shape1: ",inputs[a].tensor()->get_shape());
//   }
// };

template <typename T>
class MyMatMulOperator : public OperatorInterface<3, 1> {
 public:
  // identifiers for setting up the input tensors
  enum names_in { a, b, axis };
  // identifiers for setting up the output tensors
  enum names_out { out };

 protected:
  void compute() {
    // you can retrieve input/output tensors by its identifier
    my_matmal_kernel<T>(inputs[a].tensor(), inputs[b].tensor(), inputs[axis],
                        outputs[out].tensor());
  }
};

template <typename T>
class MyConcatOperator : public OperatorInterface<3, 1> {
 public:
  // identifiers for setting up the input tensors
  enum names_in { a, b, axis };
  // identifiers for setting up the output tensors
  enum names_out { out };

 protected:
  void compute() {
    // you can retrieve input/output tensors by its identifier
    my_concat_kernel<T>(inputs[a].tensor(), inputs[b].tensor(),
                        inputs[axis].tensor(), outputs[out].tensor());
  }
};

static const float data_a[9] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
static const float data_b[9] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
static const uint32_t data_axis[1] = {2};

static localCircularArenaAllocator<8192> meta_allocator;
static localCircularArenaAllocator<8192> ram_allocator;
// Normally there would be an additional allocator for RAM data, but it is not
// used in this tutorial.

int main(int argc, const char **argv) {
  // we only use RomTensor and BufferTensor, which use no ram, in this tutorial,
  // so we only need to setup metadata allocator, which is responsible for
  // allocating spaces storing meta data of tensors
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Tensor tensor_a = new RomTensor({1, 3, 3}, flt, data_a);
  Tensor tensor_b = new RomTensor({1, 3, 3}, flt, data_b);
  Tensor axis = new RomTensor({1}, u32, data_axis);
  // int axis = 2;

  // float* data_out = new float[9];
  float data_out[18];

  Tensor tensor_out = new BufferTensor({1, 3, 6}, flt, data_out);

  // MyAddOperator op;
  // MyMatMulOperator<float> op;
  MyConcatOperator<float> op;

  // op.set_inputs({{MyMatMulOperator<float>::a, tensor_a},
  // {MyMatMulOperator<float>::b, tensor_b}})
  //     .set_outputs({{MyMatMulOperator<float>::out, tensor_out}})
  //     .eval();

  // op.set_inputs({{MyMatMulOperator<float>::a, tensor_a},
  // {MyMatMulOperator<float>::b, tensor_b}});
  // op.set_outputs({{MyMatMulOperator<float>::out, tensor_out}});
  // op.eval();

  op.set_inputs({{MyConcatOperator<float>::a, tensor_a},
                 {MyConcatOperator<float>::b, tensor_b},
                 {MyConcatOperator<float>::axis, axis}});
  op.set_outputs({{MyConcatOperator<float>::out, tensor_out}});
  op.eval();

  // after eval(), you can read the output with () operator
  for (uint32_t i = 0; i < tensor_out->num_elems(); ++i) {
    float elem = static_cast<float>(tensor_out(i));
    printf("%ith element of output tensor: %0.1f\n", i, elem);
  }
  return 0;
}