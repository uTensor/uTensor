#ifndef UTENSOR_SDTENSOR_H
#define UTENSOR_SDTENSOR_H
#include "tensor.hpp"
#include "tensorIdxImporter.hpp"
template <class T>
class SDTensor : public Tensor {
  public:
    SDTensor(TName _name, std::string filename, uint32_t maxsize) : Tensor(_name) {
        _filename = filename;
        s->max_size = maxsize;
        cursor = 0;
        dirty = false;
    }

    SDTensor(std::initializer_list<uint32_t> l, TName _name, std::string file, uint32_t maxsize) : Tensor(_name) {
      std::vector<uint32_t> v;
      for (auto i : l) {
         v.push_back(i);
      }
      s->max_size = maxsize;
      Tensor::init<T>(v);
      _filename = file;
      cursor = 0;
      setIdxType();
      data_importer.load_data<T>(_filename, type, unit_size(), s->total_size, cursor, (T*)s->data);
      dirty = false;
    }

    SDTensor(std::vector<uint32_t> v, TName _name, std::string file, uint32_t size) : Tensor(_name) {
      s->max_size = size;
      Tensor::init<T>(v);
      _filename = file;
      cursor = 0;
      setIdxType();
      data_importer.load_data<T>(_filename, type, unit_size(), s->total_size, cursor, (T*)s->data);
      dirty = false;
    }
    void setIdxType() {
      if (std::is_same<T, unsigned char>::value) {
          type = idx_ubyte;
      } else if (std::is_same<T, char>::value) {
          type = idx_byte;
      } else if (std::is_same<T, short>::value) {
          type = idx_short;
      } else if (std::is_same<T, int>::value) {
          type = idx_int;
      } else if (std::is_same<T, float>::value) {
          type = idx_float;
      } else if (std::is_same<T, double>::value) {
          type = idx_double;
      } else {
          ERR_EXIT("idx type not supported");
      }
    }
    virtual void* read(size_t offset, size_t ele) override {
      if (ele > s->total_size) {
        ERR_EXIT("data overflow");
      }
      if (offset + ele < cursor + s->total_size) {
        //1. shared && not miss state
        //2. dirty && not miss state
        dirty = false;
        return (void *)((T*)s->data + offset - cursor);
      } else if (dirty && offset + ele >= cursor + s->total_size) { 
        //1. dirty && miss state
        data_importer.flush_data<T>(_filename, type, unit_size(), s->total_size, cursor, (T*)s->data);
        data_importer.load_data<T>(_filename, type, unit_size(), ele, offset, (T*)s->data);
        cursor = offset;
        dirty = false;
      } else if (!dirty && offset + ele >= cursor + s->total_size) {
        //1. shared && miss state
        data_importer.load_data<T>(_filename, type, unit_size(), ele, offset, (T*)s->data);
        cursor = offset;
      } 
      return (void *)((T*)s->data);
    }
    virtual void* write(size_t offset, size_t ele) override {
      if (ele > s->total_size) {
        ERR_EXIT("data overflow");
      }
      if (offset + ele < cursor + s->total_size) {
        //1. dirty && not miss state
        //2. shared && not miss state
        dirty = true;
        return (void *)((T*)s->data + offset - cursor);
      } else if (dirty && offset + ele >= cursor + s->total_size) {
        //1. dirty && miss state
        data_importer.flush_data<T>(_filename, type, unit_size(), cursor, s->total_size, (T*)s->data);
        data_importer.load_data<T>(_filename, type, unit_size(), offset, ele, (T*)s->data);
        cursor = offset;
      } else if (!dirty && offset + ele >= cursor + s->total_size) {
        //1. shared && miss state
        data_importer.load_data<T>(_filename, type, unit_size(), offset, ele, (T*)s->data);
        cursor = offset;
        dirty = true;
      }
      return (void*)((T*)s->data + offset);
    }


  // virtual void* read(size_t offset, size_t ele) override{};
  virtual uint16_t unit_size(void) override {
    return sizeof(T);
  }
  ~SDTensor() {}
 private:
  SDTensor(const SDTensor&);
  TensorIdxImporter data_importer;
  std::string _filename;
  bool dirty;
  long int cursor;
  SDTensor& operator=(const SDTensor&);
  IDX_DTYPE type;

};
#endif
