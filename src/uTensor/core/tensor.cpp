#include "tensor.hpp"

void uTensor::setName(utensor::string _name)
{
    if(name == "") {
        name = _name;
    } else {
        ERR_EXIT("Tensor %s already has a name %s\r\n", _name.c_str(), name.c_str());
    }
}
const utensor::string& uTensor::getName() const { return name; }
uTensor::~uTensor(){}

void TensorBase::initialize(const TensorShape& vec) 
{
    uint32_t ret = 0;
    shape.clear();
    for (auto ele : vec) {
        shape.push_back(ele);
        if (ret == 0) {
            ret = ele;
        } else {
            ret *= ele;
        }
    }
    total_size = ret;
}

void TensorBase::allocate(uint8_t unit_size) {
    if (total_size > cache_size) {
        data = (void*)malloc(unit_size * cache_size);
    } else {
        data = (void*)malloc(unit_size * total_size);
    }
    if (data == NULL)
        ERR_EXIT("ran out of memory for %u malloc", (unsigned int)(unit_size * total_size));
}

TensorBase::~TensorBase() {
    if (data != nullptr) {
        free(data);
        DEBUG("TensorBase memory freed..\r\n");
    }
}


void* Tensor::read(size_t offset, size_t ele) { return nullptr; }
void* Tensor::write(size_t offset, size_t ele) { return nullptr; }

Tensor::Tensor() {
    s = std::make_shared<TensorBase>();
    s->total_size = 0;
    s->cache_size = std::numeric_limits<uint32_t>::max();
    s->data = nullptr;
    setName("");
}

// returns how far a given dimension is apart
size_t Tensor::getStride(size_t dim_index) {
    unsigned int size_accm = 1;
    for (auto it = s->shape.begin() + dim_index + 1; it != s->shape.end();
            it++) {
        size_accm *= *it;
    }

    return (size_t)size_accm;
}

void Tensor::init(const TensorShape& v) {

    s->initialize(v);
    if (s->data != NULL) {
        return;
    }
    s->allocate(unit_size());
}

void Tensor::init(const TensorShape& v, const void* data) {

    s->initialize(v);
    if (s->data != NULL) {
        return;
    }
    s->data = (void *)data;
}

void Tensor::resize(const TensorShape& v) {
    uint32_t size = s->total_size;

    s->initialize(v);

    if (size == s->total_size) {
        return;
    }

    s->allocate(unit_size());
}

const TensorShape& Tensor::getShape(void) const { return s->shape; }

uint32_t Tensor::getSize(void) { return s->total_size; }

uint16_t Tensor::unit_size(void) { return 0; }

uint32_t Tensor::getSize_in_bytes(void) { return s->total_size * unit_size(); }

// returns the number of dimensions in the tensor
size_t Tensor::getDim(void) { return s->shape.size(); }


Tensor::~Tensor() {
    s = nullptr;
    DEBUG("Tensor Destructed\r\n");
}

void permuteIndexTransform::computeOutputShape(void) {
    out_stride.clear();
    if (in_shape.empty()) ERR_EXIT("input shape not set");
    if (permute.empty() || permute.size() != in_shape.size())
        ERR_EXIT("invalid permute vector");

    for (auto&& curr_axis : permute) {
        out_shape.push_back(in_shape[curr_axis]);
    }
}

size_t permuteIndexTransform::evalStride(size_t dim_index, const TensorShape& s) {
    unsigned int size_accm = 1;
    for (auto it = s.begin() + dim_index + 1; it != s.end(); it++) {
        size_accm *= *it;
    }

    return (size_t)size_accm;
}

void permuteIndexTransform::computeInputStride(void) {
    in_stride.clear();
    for (uint32_t i = 0; i < in_shape.size(); i++) {
        in_stride.push_back(evalStride(i, in_shape));
    }
}
void permuteIndexTransform::computeOutputStride(void) {
    out_stride.clear();
    for (uint32_t i = 0; i < out_shape.size(); i++) {
        out_stride.push_back(evalStride(i, out_shape));
    }
}

permuteIndexTransform::permuteIndexTransform(const TensorShape& input_shape, const std::vector<uint8_t>& permute) {
    setInputShape(input_shape);
    setPermute(permute);
    apply();
}

const std::vector<uint8_t>& permuteIndexTransform::getPermute(void) const {
    return permute;
}

void permuteIndexTransform::setPermute(const std::vector<uint8_t>& _permute) {
    permute = _permute;
    depermute.resize(permute.size());
    uint8_t i = 0;
    for (auto a : permute) {
        depermute[a] = i;
        i++;
    }
}

void permuteIndexTransform::setInputShape(const TensorShape& s) { in_shape = s; }
TensorShape permuteIndexTransform::getNewShape(void) { return out_shape; }

void permuteIndexTransform::apply(void) {
    computeOutputShape();
    computeOutputStride();
    computeInputStride();
}

size_t permuteIndexTransform::operator[](const size_t index) {
    size_t out_index = 0;
    size_t rem = index;

    for (size_t curr_dim = 0; curr_dim < in_shape.size(); curr_dim++) {
        size_t curr_stride = in_stride[curr_dim];
        out_index += (rem / curr_stride) * out_stride[depermute[curr_dim]];
        rem = rem % curr_stride;
    }

    out_index += rem;

    return out_index;
}


size_t broadcastIndexTransform::evalStride(size_t dim_index, const TensorShape& s) {
    unsigned int size_accm = 1;
    for (auto it = s.begin() + dim_index + 1; it != s.end(); it++) {
        size_accm *= *it;
    }

    return (size_t)size_accm;
}

void broadcastIndexTransform::computeSStride(void) {
    s_stride.clear();
    for (uint32_t i = 0; i < s_shape.size(); i++) {
        s_stride.push_back(evalStride(i, s_shape));
    }
}
void broadcastIndexTransform::computeLStride(void) {
    l_stride.clear();
    for (uint32_t i = 0; i < l_shape.size(); i++) {
        l_stride.push_back(evalStride(i, l_shape));
    }
}

void broadcastIndexTransform::sortShape(const TensorShape& a, const TensorShape& b) {
    if(a.size() > b.size()) {
        l_shape = a;
        s_shape = b;
    } else if(a.size() < b.size()) {
        l_shape = b;
        s_shape = a;
        swap_flag = true;
    } else {
        auto it = std::find(a.begin(), a.end(), 1);
        if (it == a.end()) {
            l_shape = a;
            s_shape = b;
        } else {
            l_shape = b;
            s_shape = a;
            swap_flag = true;
        }
    }
}

void broadcastIndexTransform::checkShape(void) {
    if(l_shape.size() < s_shape.size()) ERR_EXIT("cannot boardcast to fewer dimensions");
    for(int i = 0; i < (int) l_shape.size(); i++) {
        int small_i = i - (l_shape.size() - s_shape.size());
        if(small_i < 0) continue;
        if(l_shape[i] != s_shape[small_i] && s_shape[small_i] != 1) ERR_EXIT("ValueError: frames are not aligned");
        if(l_shape[i] < s_shape[small_i]) ERR_EXIT("Only single target broadcast is supported");
    }
}

broadcastIndexTransform::broadcastIndexTransform(const TensorShape& _l_shape, const TensorShape& _s_shape) {
    swap_flag = false;
    sortShape(_l_shape, _s_shape);
    checkShape();
    apply();
}

void broadcastIndexTransform::apply(void) {
    //computeOutputShape();
    computeLStride();
    computeSStride();
}

const TensorShape& broadcastIndexTransform::getOutputShape(void) const {
    return l_shape;
}

bool broadcastIndexTransform::is_swaped(void) {
    return swap_flag;
}

size_t broadcastIndexTransform::operator[](const size_t linear_index) {
    size_t out_index = 0;
    size_t rem = linear_index;
    const size_t d_dim = l_shape.size() - s_shape.size();
    size_t s_dim;

    for (size_t curr_dim = 0; curr_dim < l_shape.size(); curr_dim++) {
        size_t curr_stride = l_stride[curr_dim];

        if(l_shape.size() - curr_dim <= s_shape.size()) {
            size_t curr_l_index = (rem / curr_stride);
            s_dim = curr_dim - d_dim;
            size_t curr_s_index = (curr_l_index % s_shape[s_dim]);
            out_index += curr_s_index * s_stride[s_dim];
        }

        rem = rem % curr_stride;
    }

    out_index += (rem % s_stride[s_dim]);

    // ///NT: DEBUG CODE
    // int sum = 0;
    // for(auto i:l_shape) {
    //   sum += i;
    // }
    // if((int)out_index > sum) {
    //   ERR_EXIT("index out of range, linear_index: %d, sum: %d, out_index: %d", linear_index, sum, out_index);
    // }
    // ///

    return out_index;
}
