typedef utensor::string OpName;
typedef std::unordered_map<TName, Tensor*> TensorNamePtrMap;
 
class TensorRecord {
    public:
        unsigned char ref_count = 0;
        Tensor* t
        TensorRecord(Tensor* _t, unsigned char _ref_count) :
            t(_t) {}
};

class TensorShape {
        TensorShape(uint16_t shape) : _shape({shape, 0, 0, 0}), _num_dims(1) {}
        TensorShape(uint16_t shape[1]) : _shape({shape, 0, 0, 0}), _num_dims(1) {}
        TensorShape(uint16_t shape[2]) : _shape({shape[0],shape[1], 0, 0}), _num_dims(2) {}
        TensorShape(uint16_t shape[3]) : _shape({shape[0], shape[1], shape[2], 0}), _num_dims(3) {}
        TensorShape(uint16_t shape[4]) : _shape(shape), _num_dims(4) {}

        uint16_t operator[] (int i) const { return _shape[i]; /* Do additional checks*/}
        uint16_t& operator[](int i) { return _shape[i]; } // Maybe handle update case
        void update_dims() { 
            for(int i = 0; i < 4; i++){
                if(_shape[i] && (i+1) < _num_dims)
                    _num_dims = i+1;
                else if(!_shape[i] && (i+1) > _num_dims)
                    _num_dims = i+1;
            }
        }
        uint16_t get_linear_size() const {

            uint16_t sum = 0;
            for(int i = 0; i < _num_dims; i++){
                sum += _shape[i];
            }
        }
    private:
        uint16_t _shape[4];
        uint8_t  _num_dims;
}
// Do something to remember current type
enum ttype: uint8_t {
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            float,
            undefined
};
// Need to figure out way of defering reference until after lefthand assignment
class IntegralValue {
    void* p;
    public:

        // Explicit
        IntegralValue(void* p) : p(p) {}
        IntegralValue(const uint8_t& u) { *reinterpret_cast<uint8_t*>(p) = u; }
        IntegralValue(const int8_t& u) { *reinterpret_cast<int8_t*>(p) = u; }
        IntegralValue(const uint16_t& u) { *reinterpret_cast<uint16_t*>(p) = u; }
        IntegralValue(const int16_t& u) { *reinterpret_cast<int16_t*>(p) = u; }
        IntegralValue(const uint32_t& u) { *reinterpret_cast<uint32_t*>(p) = u; }
        IntegralValue(const int32_t& u) { *reinterpret_cast<int32_t*>(p) = u; }

        //IntegralValue& operator=(void* _p) { p = _p; }

        operator uint8_t   ( ) const { return static_cast<uint8_t> ( *reinterpret_cast<uint8_t*>  ( p)); }
        operator uint8_t&  ( )       { return static_cast<uint8_t&> ( *reinterpret_cast<uint8_t*>  ( p)); }
        operator int8_t    ( ) const { return static_cast<int8_t> ( *reinterpret_cast<int8_t*>   ( p)); }
        operator int8_t&   ( )       { return static_cast<int8_t&> ( *reinterpret_cast<int8_t*>   ( p)); }
        
        operator uint16_t  ( ) const { return static_cast<uint16_t> ( *reinterpret_cast<uint16_t*> ( p)); }
        operator uint16_t& ( )       { return static_cast<uint16_t&> ( *reinterpret_cast<uint16_t*> ( p)); }
        operator int16_t   ( ) const { return static_cast<int16_t> ( *reinterpret_cast<int16_t*>  ( p)); }
        operator int16_t&  ( )       { return static_cast<int16_t&> ( *reinterpret_cast<int16_t*>  ( p)); }

        operator uint32_t  ( ) const { return static_cast<uint32_t> ( *reinterpret_cast<uint32_t*> ( p)); }
        operator uint32_t& ( )       { return static_cast<uint32_t&> ( *reinterpret_cast<uint32_t*> ( p)); }
        operator int32_t   ( ) const { return static_cast<int32_t> ( *reinterpret_cast<int32_t*>  ( p)); }
        operator int32_t&  ( )       { return static_cast<int32_t&> ( *reinterpret_cast<int32_t*>  ( p)); }

        operator float   ( ) const { return static_cast<float> ( *reinterpret_cast<float*>  ( p)); }
        operator float&  ( )       { return static_cast<float&> ( *reinterpret_cast<float*>  ( p)); }

};

// template<typename Allocator=utensor::DefaultTensorMetaDataAllocator>
class TensorBase {
public:
    TensorBase(){
        utensor::Context::get_default_context().register(*this);
    }
    virtual ~TensorBase(){}

    // Allocate the tensor metadata on a different heap from the data scratch pads

    virtual void* operator new(size_t sz) { 
        void* p = utensor::Context::DefaultTensorMetaDataAllocator::allocate(sz); 
        return p;
    }
    virtual void operator delete(void* p) {
        utensor::Context::DefaultTensorMetaDataAllocator::deallocate(p);
    }
};

class TensorInterface : public TensorBase {
    protected:
        virtual void* read(uint32_t linear_index) const = 0; // Handle to the data
        virtual void* write(uint32_t linear_index) = 0
    public:
        ttype get_type() const { return _type; }
        TensorShape& get_shape() { return _shape; }
        TensorInterface() : TensorBase(), _shape(0), _type(undefined) {}
        TensorInterface(TensorShape _shape, ttype _type) : TensorBase(), _shape(_shape), _type(_type) {}
        virtual ~TensorInterface() {};

        // Can access Tensors like
        // mTensor(1) = 5, mTensor(2,2) = 5, etc.
        const IntegralValue operator()(uint16_t i, uint16_t j = 0, uint16_t k = 0, uint16_t l = 0){
            // Add shape checks here
            return read(_shape.linear_index(i, j, k, l));
        }
        IntegralValue& operator()(uint16_t i, uint16_t j = 0, uint16_t k = 0, uint16_t l = 0){
            // Add shape checks here
            return write(_shape.linear_index(i,j,k,l));
        }
        virtual void resize(TensorShape new_shape) = 0;

    private:
        TensorShape _shape;
        ttype _type; // Maybe make this const
};

class utensor::AllocatorInterface {

    protected:
        virtual void _bind(void* ptr, utensor::Tensor* hndl) = 0;
        virtual void _unbind(void* ptr, utensor::Tensor* hndl) = 0;
        virtual bool _is_bound(void* ptr, utensor::Tensor* hndl) = 0;
        virtual bool _has_handle(utensor::Tensor* hndl) = 0;

    public:
        void update_hndl(Tensor& h, utensor::Tensor* new_t_ptr) {
            h._ptr = new_t_ptr;
        }

        void bind(void* ptr, utensor::Tensor* hndl) { 
            if (!has_hndl(hndl))
                ERROR("Allocator does not contain reference to handle");

            if (is_bound(ptr, hndl)){
                ERROR("Cannot rebind Handles without unbinding");
                exit(-1)
            }
            _bind(ptr, hndl);
        }
        void unbind(void* ptr, utensor::Tensor* hdnl) {
            if (!is_bound(ptr, hndl)){
                ERROR("Cannot unbind unbound Handles");
                exit(-1)
            }
            _unbind(ptr, hndl);            
        }
        bool is_bound(void* ptr, utensor::Tensor* hndl) {
            return _is_bound(ptr, hndl);
        }

        virtual bool can_rebalance() = 0;
        virtual size_t available() = 0;
        virtual bool rebalance() = 0; // KEY. This call updates all the Tensor data references
        
        void* allocate(size_t sz) = 0;
        void deallocate(void* ptr) = 0

};

template<size_t size>
class utensor::localArenaAllocator : public AllocatorInterface {
    // stuff
};
// Note not actually complete
template<size_t sz>
class utensor::ArenaAllocator {
    private:
        static localArenaAllocator<sz> _allocator;
    public:
        static void* allocate(size_t size) { 
            if (size > _allocator.available())
                return NULL;

            void* p = _allocator.allocate(size);
            if (p == NULL)
                _allocator.rebalance(); 
        }
        static void  deallocate(void* p) { ... }
        static void  bind(void* ptr, utensor::Tensor* hndl) {
            _allocator.bind(ptr, hndl);
        }

};

using utensor::DefaultTensorMetaDataAllocator = utensor::ArenaAllocator<512>;
using utensor::DefaultRamTensorAllocator = utensor::ArenaAllocator<4096>;

// Tensors also appear on the same heap as the Tensor metadata. This way we can move tensors around and delete them without affecting user code
//template <typename Allocator=utensor::DefaultTensorMetaDataAllocator>
class Tensor {
    private:
        utensor::TensorInterface* _ptr;
        Tensor(const Tensor& that) {} // Cannot copy Tensors, must pass by reference

    public:  
        utensor::TensorInterface* operator->(0) { return _ptr; }
        Tensor(utensor::TensorInterface* ptr) : _ptr(ptr) {
            Context::DefaultTensorMetaDataAllocator::bind(this, ptr);
        }
        // Add some bits to make the interface nicer to the user

        // Force everything to be on the utensor allocator
        void* operator new(size_t sz) { // Have to delegate this size from tensors somehow + sizeof(Tensor)
            void* p = Context::DefaultTensorMetaDataAllocator::allocate(sz); 
            return p;
        }
        void operator delete(void* p) {
            Context::DefaultTensorMetaDataAllocator::deallocate(p);
        }

        // KEY BIT
        friend class utensor::AllocatorInterface;
};

// sizeof(NamedTensorReference) == 8 bytes
class NamedTensorReference {
    public:
    const utensor::string& name; //Fixed
    utensor::Tensor& tensor;     //Modifiable
    
    NamedTensorReference(const utensor::string& name, utensor::Tensor& tensor) : name(name), tensor(tensor) {
        Context& ctx = Context::get_default_context();
        ctx.push(*this);
    }
    
};

// Same as Named Tensor but not registered in the context class 
class SimpleNamedTensor {
    public:
    const utensor::string& name; //Fixed
    utensor::Tensor& tensor;     //Modifiable
    
    SimpleNamedTensor(const utensor::string& name, utensor::Tensor& tensor) : name(name), tensor(tensor) {}
};

// Tensor maps are fixed size to force input output mismatched errors
class TensorMapInterface {
public:
    virtual SimpleNamedTensor& operator[](const utensor::string& name) = 0;
    virtual const SimpleNamedTensor& operator[](const utensor::string& name) const = 0;
    static SimpleNamedTensor not_found(utensor::string("NotFound"), static_cast<utensor::Tensor>(NULL));
};

template<size_t size>
class FixedTensorMap : public TensorMapInterface{
public:
    TensorMap(SimpleNamedTensor map[size]) : _map[map] {}
    virtual ~TensorMap() {}
    SimpleNamedTensor& operator[](const utensor::string& name){
        for(int i = 0; i < size; i++){
            if(name == _map[i].name)
                return _map[i];
        }
        return TensorMapInterface::not_found;
    }
    const SimpleNamedTensor& operator[](const utensor::string& name) const {
        for(int i = 0; i < size; i++){
            if(name == _map[i].name)
                return _map[i];
        }
        return TensorMapInterface::not_found;
    }
private:
    SimpleNamedTensor _map[size];
};


// Operators do not go on the heap
class OperatorBase {
protected:
    TensorMapInterface& inputs;
    TensorMapInterface& outputs;
public:
    utensor::string op_name;
public:
    OperatorBase(TensorMapInterface* inputs, TensorMapInterface* outputs) : inputs(*inputs), outputs(*outputs) {
        Context& ctx = Context::get_default_context();
        ctx.push_op_tensors(*this, inputs);
        ctx.push_op_tensors(*this, outputs);
    }
    // The preferred interface
    OperatorBase(TensorMapInterface* inputs) : inputs(*inputs) {
        Context& ctx = Context::get_default_context();
        ctx.push_op_tensors(*this, inputs);
    }
    OperatorBase() {}
    void set_inputs(TensorMapInterface* inputs) {
        this->inputs = *inputs; 
        Context& ctx = Context::get_default_context();
        ctx.push_op_tensors(*this, inputs);
    }
    void set_outputs(TensorMapInterface* outputs) {
        this->outputs = *outputs; 
        Context& ctx = Context::get_default_context();
        ctx.push_op_tensors(*this, outputs);
    }
    virtual ~OperatorBase() {
        Context& ctx = Context::get_default_context();
        ctx.pop_op_tensors(*this, inputs); // Inputs are no longer needed
    }

protected:
    friend class Context;
    virtual void compute() = 0;
};

void add_kernel(Tensor& a, Tensor& b, Tensor& c){
    // Decide on c shape
    for (int i = 0; i < c.size(); i++)
        c[i] = a[i] + b[i];
}

class AddOperator {
public:
    static enum names: uint8_t {a, b, c};
    AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) : OperatorBase(inputs, outputs) {}

protected:
    virtual void compute() {
        add_kernel(outputs[c], inputs[a], inputs[b]);
    }

}

class Context {
    private:
        Maintain List of op tensors, NamedTensorReferences, and root tensors
        Context();

    public:
        static utensor::DefaultTensorMetaDataAllocator DefaultTensorMetaDataAllocator<512>;
    public:
        static Context* get_default_context() { static Context ctx; return *ctx;}
//         template<class T, typename... Args>
//         Tensor* add(TName&& _name, unsigned char&& _ref_count, Args&&... args) {
//             //pooling can be implemented here
//             Tensor* t = new T(std::forward<Args>(args)...);
//             t->setName(_name);
//             tTable[_name] = TensorRecord(t, _ref_count);
//             return t;
//         }
//         Tensor*& operator[](TName name);  //non-existing tensor: returns Tensor*& but set Tensor* to null
        
        // Op names are just used for debugging
        void invoke(operator *op);  //persistent op exists in heap
        void invoke(operator &op);  //intermediate ops exists on stack
        //void invoke(operator &&op, TensorMap _map, OpName _name);  //intermediate ops mamanged by stack
        void gc({});  //decrease ref count of used tensors and perform deletion
};

void example_model(RamTensor<int8_t> input){
    // Context is opaque and will handle the destruction and reference counting
    Context& ctx = Context::get_default_context();

    Tensor& A = new Tensor(new RomTensor<int8_t(SREF_A_SHAPE, SREF_A_DATA)):
    Tensor& B = new Tensor(new RomTensor<int8_t(SREF_B_SHAPE, SREF_B_DATA)):
    Tensor& C = new Tensor(new RamTensor<int8_t>());

    AddOperator add_AB().set_inputs({{AddOperator::a, A}, {AddOperator::b, B}}).set_outputs({{AddOperator::c, C}});

    ctx.invoke(add_AB); 
    
}
