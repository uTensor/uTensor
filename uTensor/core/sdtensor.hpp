#ifndef UTENSOR_SDTENSOR_H
#define UTENSOR_SDTENSOR_H
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/vm.hpp"

#ifdef MBED_CONF_APP_DEBUG_MSG
  #define tmpprefix "/fs/tmp/"
#else
  #define tmpprefix "/tmp/"
#endif

///NT: FIXME: count overflow
static string getTmpName(void) {
  static uint32_t count = 0;
  return std::to_string(count++);
}

template <class T>
class SDTensor : public Tensor {
  public:
    SDTensor( uint32_t cachesize) : Tensor() {
        string file = tmpprefix + getTmpName();
        _filename = file;
        mem.createFile(_filename);
        s->cache_size = cachesize;
        cursor = 0;
        dirty = false;
    }

    SDTensor(std::initializer_list<uint32_t> l, uint32_t cachesize) : Tensor() {
      std::vector<uint32_t> v;
      for (auto i : l) {
         v.push_back(i);
      }
      s->cache_size = cachesize;
      Tensor::init(v);
      string file = tmpprefix + getTmpName();
      _filename = file;
      mem.createFile(_filename);
      cursor = 0;
      dirty = false;
    }

    SDTensor(const std::vector<uint32_t>& v, uint32_t cachesize) : Tensor() {
      s->cache_size = cachesize;
      Tensor::init(v);
      string file = tmpprefix + getTmpName();
      _filename = file;
      mem.createFile(_filename);
      cursor = 0;
      dirty = false;
    }
    virtual void* read(size_t offset, size_t ele) override {
      if (ele > s->total_size) {
        ERR_EXIT("data overflow");
      }
      if (offset + ele <= cursor + s->cache_size && offset + ele >= cursor) {
        //1. shared && not miss state
        //2. dirty && not miss state
        dirty = false;
        return (void *)((T*)s->data + offset - cursor);
      } else if (dirty && (offset + ele > cursor + s->cache_size || offset + ele < cursor)) {
        //1. dirty && miss state
        mem.flush_data<T>(_filename, unit_size(), s->cache_size, s->total_size, cursor, (T*)s->data);
        mem.load_data<T>(_filename, unit_size(), s->cache_size, s->total_size, offset, (T*)s->data);
        cursor = offset;
        dirty = false;
      } else if (!dirty && (offset + ele > cursor + s->cache_size || offset + ele < cursor)) {
        //1. shared && miss state
        mem.load_data<T>(_filename, unit_size(), s->cache_size, s->total_size, offset, (T*)s->data);
        cursor = offset;
      }
      return (void *)((T*)s->data);
    }
    virtual void* write(size_t offset, size_t ele) override {
      if (ele > s->total_size) {
        ERR_EXIT("data overflow");
      }
      if (offset + ele <= cursor + s->cache_size && offset + ele >= cursor) {
        //1. dirty && not miss state
        //2. shared && not miss state
        dirty = true;
        return (void *)((T*)s->data + offset - cursor);
      } else if (dirty && (offset + ele > cursor + s->cache_size || offset + ele < cursor)) {
        //1. dirty && miss state
        mem.flush_data<T>(_filename, unit_size(), s->cache_size, s->total_size, cursor, (T*)s->data);
        mem.load_data<T>(_filename, unit_size(), s->cache_size, s->total_size, offset, (T*)s->data);
        cursor = offset;
      } else if (!dirty && (offset + ele > cursor + s->cache_size || offset + ele < cursor)) {
        //1. shared && miss state
        mem.load_data<T>(_filename, unit_size(), s->cache_size, s->total_size, offset, (T*)s->data);
        cursor = offset;
        dirty = true;
      }
      return (void*)((T*)s->data);
    }
    void resize(const std::vector<uint32_t>& v) override {
        Tensor::resize(v);
        initCache();
    }
    void initCache() {
        mem.load_data<T>(_filename, unit_size(), s->cache_size, s->total_size, 0, (T*)s->data);
    }

    virtual void deFocus() override{
        mem.flush_data<T>(_filename, unit_size(), s->cache_size, s->total_size, cursor, (T*)s->data);
        dirty = false;
    }

    vm getVM() {
        return mem;
    }

  // virtual void* read(size_t offset, size_t ele) override{};
  virtual uint16_t unit_size(void) override {
    return sizeof(T);
  }

  ~SDTensor() {
  }
 private:
  SDTensor(const SDTensor&);
  vm mem;
  std::string _filename;
  bool dirty;
  uint32_t cursor;
  SDTensor& operator=(const SDTensor&);

};
#endif
