#include <cstring>
#include <iostream>

#include "gtest/gtest.h"
#include "uTensor.h"

using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

const uint8_t s_a[4] = {1, 2, 3, 4};
const uint8_t s_b[4] = {5, 6, 7, 8};
const uint8_t s_c[4] = {5, 6, 7, 8};
const uint8_t s_d[4] = {1, 1, 1, 1};
// out = a*b + c + d

const uint8_t s_out_ref[4] = {19 + 5 + 1, 22 + 6 + 1, 43 + 7 + 1, 50 + 8 + 1};

const size_t my_model_num_inputs = 1;
const size_t my_model_num_outputs = 1;

// This example Model allocates all RAM tensors once, and keeps them allocated
// for the lifetime of the model out = a*b + c + d
class MyModel
    : public ModelInterface<my_model_num_inputs, my_model_num_outputs> {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };
  MyModel();

  // Inherits public interface
  // eval();
  // set_inputs(...);
  // set_outputs(...);
 protected:
  virtual void compute();

 private:
  // ROM Tensors
  Tensor b;
  Tensor c;
  Tensor d;

  // RAM tensors
  // Can probably come up with some form of a naming scheme around this
  Tensor mult_1_out;
  Tensor add_1_out;

  // Operators
  // Note: only need one instance of each since we can set inputs in the compute
  // call
  MatrixMultOperator<uint8_t> mult;
  AddOperator<uint8_t> add;

  // Memory Allocators
  localCircularArenaAllocator<512> meta_allocator;
  localCircularArenaAllocator<64> ram_allocator;
};

MyModel::MyModel() {
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  b = new RomTensor({2, 2}, u8, s_b);
  c = new RomTensor({2, 2}, u8, s_c);
  d = new RomTensor({2, 2}, u8, s_d);

  // Ram Tensors are temporary values
  mult_1_out = new RamTensor({2, 2}, u8);
  add_1_out = new RamTensor({2, 2}, u8);
}
void MyModel::compute() {
  // First update the default context to this model in case multiple models are
  // being run
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  // First Multiply
  // mult_1_out = input*b;
  mult.set_inputs({{MatrixMultOperator<uint8_t>::a, inputs[input].tensor()},
                   {MatrixMultOperator<uint8_t>::b, b}})
      .set_outputs({{MatrixMultOperator<uint8_t>::c, mult_1_out}})
      .eval();

  // First Add
  // output = mult_1_out + c;
  add.set_inputs(
         {{AddOperator<uint8_t>::a, mult_1_out}, {AddOperator<uint8_t>::b, c}})
      .set_outputs({{AddOperator<uint8_t>::c, add_1_out}})
      .eval();

  // Second Add
  // output = add_1_out + d
  add.set_inputs(
         {{AddOperator<uint8_t>::a, add_1_out}, {AddOperator<uint8_t>::b, d}})
      .set_outputs({{AddOperator<uint8_t>::c, outputs[output].tensor()}})
      .eval();
}

MyModel myModel;

TEST(Model, example1) {
  // Userspace owns the buffer in a buffer tensor
  uint8_t* a_buffer = new uint8_t[2 * 2];
  uint8_t* out_buffer = new uint8_t[2 * 2];
  Tensor a = new /*const*/ BufferTensor({2, 2}, u8, a_buffer);
  Tensor out = new /*const*/ BufferTensor({2, 2}, u8, out_buffer);

  // Go ahead and copy example input to a_buffer
  // Simulate data write
  for (int i = 0; i < 4; i++) {
    a_buffer[i] = s_a[i];
  }

  myModel.set_inputs({{MyModel::input, a}})
      .set_outputs({{MyModel::output, out}})
      .eval();

  // Compare results using buffers
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(out_buffer[i], s_out_ref[i]);
  }

  // Compare results using the Tensor
  Tensor out_ref = new RomTensor({2, 2}, u8, s_out_ref);
  TensorShape& out_shape = out->get_shape();
  for (int i = 0; i < out_shape[0]; i++) {
    for (int j = 0; j < out_shape[1]; j++) {
      // Just need to cast the output
      EXPECT_EQ((uint8_t)out(i, j), (uint8_t)out_ref(i, j));
    }
  }

  delete[] a_buffer;
  delete[] out_buffer;
}
