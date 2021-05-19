#include "arenaAllocator.hpp"
namespace uTensor {

DEFINE_EVENT(MetaHeaderNotFound);
DEFINE_EVENT(localCircularArenaAllocatorRebalancing);
DEFINE_EVENT(localCircularArenaAllocatorConstructed);
DEFINE_ERROR(InvalidBoundRegionState);
DEFINE_ERROR(InvalidAlignmentAllocation);
DEFINE_ERROR(MetaHeaderNotBound);

localCircularArenaAllocatorBase::MetaHeader::MetaHeader()
    : meta_data(BLOCK_INACTIVE & BLOCK_ZERO_LENGTH),
      hndl(nullptr),
      _d(nullptr) {}
localCircularArenaAllocatorBase::MetaHeader::MetaHeader(uint32_t sz)
    : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr), _d(nullptr) {}
localCircularArenaAllocatorBase::MetaHeader::MetaHeader(uint32_t sz, uint8_t* d)
    : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr), _d(d) {}
void localCircularArenaAllocatorBase::MetaHeader::set_active() { meta_data |= BLOCK_ACTIVE; }
void localCircularArenaAllocatorBase::MetaHeader::set_inactive() { meta_data &= BLOCK_INACTIVE; }
void localCircularArenaAllocatorBase::MetaHeader::set_hndl(Handle* handle) { hndl = handle; }
void localCircularArenaAllocatorBase::MetaHeader::set_d(uint8_t* d) { _d = d; }
void localCircularArenaAllocatorBase::MetaHeader::set_len(uint32_t sz) {
  meta_data &= MSB_SET;  // Clear all size bits
  meta_data |= (BLOCK_LENGTH_MASK & sz);
}
uint32_t localCircularArenaAllocatorBase::MetaHeader::get_len() const { return meta_data & BLOCK_LENGTH_MASK; }
bool localCircularArenaAllocatorBase::MetaHeader::is_active() const { return (meta_data & MSB_SET) == BLOCK_ACTIVE; }
bool localCircularArenaAllocatorBase::MetaHeader::is_bound() const { return (hndl != nullptr); }
bool localCircularArenaAllocatorBase::MetaHeader::has_handle(Handle* target) const {
  return is_active() && (hndl == target);
}
bool localCircularArenaAllocatorBase::MetaHeader::is_used() const { return is_active() && (get_len() > 0); }


// Return the amount of free space at the tail
uint32_t localCircularArenaAllocatorBase::tail_capacity(){};

size_t localCircularArenaAllocatorBase::find_header_associated_w_ptr(void* ptr) const {
  size_t i = 0;
  for (auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++) {
    if (hdr_i->_d == ptr) return i;
  }
  return i;
}
// This is just for reference
localCircularArenaAllocatorBase::MetaHeader& localCircularArenaAllocatorBase::_read_header(void* ptr) {
  static MetaHeader not_found;
  // First check if ptr in bounds
  if (ptr < begin() || ptr > end()) {
    // ERROR
    Context::get_default_context()->throwError(new OutOfMemBoundsError);
  }
  for (auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++) {
    if (hdr_i->_d == ptr) return *hdr_i;
  }
  // ERROR
  Context::get_default_context()->notifyEvent(MetaHeaderNotFound());
  return not_found;
}

void* localCircularArenaAllocatorBase::attempt_to_reuse_inactive_region(size_t sz) {
  uint8_t* loc = nullptr;
   for (auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++) {
     if (!hdr_i->is_active() && hdr_i->get_len() >= sz) {
       MetaHeader& hdr = *hdr_i;
       // Handle alignment
       void* aligned_loc = (void*)hdr._d;
       size_t space_change = hdr.get_len();
       aligned_loc =
           std::align(alignof(uint8_t*), sz, aligned_loc, space_change);
       if(aligned_loc == nullptr){
         Context::get_default_context()->throwError(new InvalidAlignmentAllocation);
       }
       hdr.set_active();
       // hdr.set_len(sz + hdr.get_len() - space_change);
       hdr.set_len(sz);
       hdr.set_hndl(nullptr);
       hdr.set_d((uint8_t*)aligned_loc);
       loc = (uint8_t*)aligned_loc;

       // Update capacity
       //capacity -= hdr.get_len();

       return (void*)loc;
     }
   }
   return nullptr;
}

inline uint8_t* localCircularArenaAllocatorBase::begin() const { return _buffer; }
inline const uint8_t* localCircularArenaAllocatorBase::end() const { return _buffer + size; }
inline size_t localCircularArenaAllocatorBase::_get_size() const { return size; }

void localCircularArenaAllocatorBase::_bind(void* ptr, Handle* hndl) {
  MetaHeader& hdr = _read_header(ptr);
  // Check if region is active
  if (!hdr.is_active()) {
    // ERROR
    Context::get_default_context()->throwError(new InvalidBoundRegionState);
  }
  hdr.set_hndl(hndl);
}

 void localCircularArenaAllocatorBase::_unbind(void* ptr, Handle* hndl) {
  MetaHeader& hdr = _read_header(ptr);
  if (!hdr.is_active()) {
    // ERROR
    Context::get_default_context()->throwError(new InvalidBoundRegionState);
  }
  // teehee
  update_hndl(hndl, nullptr);
  hdr.set_hndl(nullptr);
  //_bind(ptr, nullptr);
}

 bool localCircularArenaAllocatorBase::_is_bound(void* ptr, Handle* hndl) {
  MetaHeader hdr = _read_header(ptr);
  // Check if region is active
  if (!hdr.is_active()) {
    // ERROR
    Context::get_default_context()->throwError(new MetaHeaderNotBound);
  }
  return hdr.is_bound() && (hdr.hndl == hndl);
  ;
}

 bool localCircularArenaAllocatorBase::_has_handle(Handle* hndl) {
  for (auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++) {
    if (hdr_i->has_handle(hndl)) return true;
  }
  return false;
}

 void* localCircularArenaAllocatorBase::_allocate(size_t sz) {
  uint8_t* loc = nullptr;
  // If make this capacity then have possibility of filling up
  if (sz > _get_size()) {
    // ERROR
    Context::get_default_context()->throwError(new OutOfMemError);
    return nullptr;
  }
  // if(sz > ( end() - (cursor + sizeof(MetaHeader)))){
  if (sz > available()) {
    // Allocate at beginning
    // Rebalance to make it less likely to overwrite a region
    // Overwriting allocated regions is a valid operation as long as the
    // overwritten regions are invalidated
    rebalance();
    // If still dont have space, error out
    if (sz > available()) {
      Context::get_default_context()->throwError(new OutOfMemError);
      return nullptr;
    }
  }

  // First check to see if we have space in a previously allocated area
  // TODO: if this region is smaller split it and add another header to the
  // table
  void* reallocated = attempt_to_reuse_inactive_region(sz);
  if(reallocated){
    return reallocated;
  }
  if (sz > (end() - reinterpret_cast<uint8_t*>(cursor))){
    rebalance();

  }
  // Otherwise allocate at the end
  MetaHeader hdr;
  // Handle alignment
  void* aligned_loc = (void*)cursor;
  size_t space_change = available();
  aligned_loc = std::align(alignof(uint8_t*), sz, aligned_loc, space_change);
  if(aligned_loc == nullptr){
    Context::get_default_context()->throwError(new InvalidAlignmentAllocation);
  }
  hdr.set_active();
  // hdr.set_len(sz + available() - space_change);
  hdr.set_len(sz);
  hdr.set_hndl(nullptr);
  hdr.set_d((uint8_t*)aligned_loc);
  _headers.push_back(hdr);
  loc = (uint8_t*)aligned_loc;
  //cursor += hdr.get_len() + available() - space_change;
  cursor = reinterpret_cast<uint8_t*>(aligned_loc);
  cursor += hdr.get_len();

  // Update capacity
  capacity -= hdr.get_len();

  return (void*)loc;
}

 void localCircularArenaAllocatorBase::_deallocate(void* ptr) {
  if (ptr) {
    MetaHeader& hdr = _read_header(ptr);
    if (hdr.is_bound()) {
      _unbind(ptr, hdr.hndl);
    }
    hdr.set_inactive();
    hdr.set_hndl(nullptr);  // cleanup
    //capacity += hdr.get_len();
    // Do not update the size of the header
  }
}

localCircularArenaAllocatorBase::localCircularArenaAllocatorBase(uint8_t* buffer, size_t size) : _buffer(buffer), size(size), capacity(size) {
  Context::get_default_context()->notifyEvent(
      localCircularArenaAllocatorConstructed());
  cursor = begin();
}
 localCircularArenaAllocatorBase::~localCircularArenaAllocatorBase() {
  
}

/** This implementation of rebalance shifts all allocated chunks to the end of
 * the buffer and inserts an inactive region at the start. note: cursor gets
 * moved to begin() note: unbound regions get wiped
 */
// TODO Check to make sure updated locations are still aligned
 bool localCircularArenaAllocatorBase::rebalance() {
  Context::get_default_context()->notifyEvent(
      localCircularArenaAllocatorRebalancing());
  // Clear all unbound entries
  for (auto hdr_i = _headers.rbegin(); hdr_i != _headers.rend(); hdr_i++) {
    if (!hdr_i->is_bound()) {
      hdr_i->set_inactive();
    }
  }
  // Sort by activity (shifts unbound entries to the end)
  std::sort(_headers.begin(), _headers.end(),
            [](const MetaHeader& a, const MetaHeader& b) {
              return a.is_active() > b.is_active();
            });

  int pop_count = 0;
  for (auto hdr_i = _headers.rbegin(); hdr_i != _headers.rend(); hdr_i++) {
    if (hdr_i->is_active()) {
      break;
    }
    capacity += hdr_i->get_len();
    pop_count++;
  }
  // Remove all unbound
  // Makes the allocator have a cold start
  for (int i = 0; i < pop_count; i++) {
    _headers.pop_back();
  }

  // Headers now only has the bound regions
  // Sort by region
  std::sort(
      _headers.begin(), _headers.end(),
      [](const MetaHeader& a, const MetaHeader& b) { return a._d < b._d; });

  uint8_t tmp;
  cursor = begin();
  void* aligned_loc;
  //size_t space_change;

  for (auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++) {
    aligned_loc = (void*)cursor;
    //size_t space_change = available();
    size_t space_change = (hdr_i->_d - cursor) + hdr_i->get_len();
    aligned_loc = std::align(alignof(uint8_t*), hdr_i->get_len(), aligned_loc,
                             space_change);
    if(aligned_loc == nullptr){
      Context::get_default_context()->throwError(new InvalidAlignmentAllocation);
    }

    // Shift the data
    for (size_t i = 0; i < hdr_i->get_len(); i++) {
      tmp = hdr_i->_d[i];
      reinterpret_cast<uint8_t*>(aligned_loc)[i] = tmp;
    }

    // Update header
    // hdr_i->set_len(sz + available() - space_change);
    hdr_i->set_d((uint8_t*)aligned_loc);
    update_hndl(hdr_i->hndl, hdr_i->_d);
    //cursor += hdr_i->get_len() + available() - space_change;
    cursor = reinterpret_cast<uint8_t*>(aligned_loc);
    cursor += hdr_i->get_len();
  }
  capacity = end() - cursor;
  return true;
}

 size_t localCircularArenaAllocatorBase::available() { return capacity; }

 void localCircularArenaAllocatorBase::clear() {
  // TODO deallocate and invalidate all references
  // reset to default state
  memset(_buffer, 0, size);
  cursor = begin();
  capacity = _get_size();
}

// Check to see if pointer exists in memory space and is valid
bool localCircularArenaAllocatorBase::contains(void* p) const {
  if (!((p >= begin()) && (p < end()))) {
    return false;
  }
  for (auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++) {
    if (hdr_i->_d == p) return hdr_i->is_used();
  }
  return false;
  // MetaHeader hdr = _read_header(p);
  // return hdr.is_used();
}

// Testing bits, attribute out later
uint32_t localCircularArenaAllocatorBase::internal_header_unit_size() const { return sizeof(MetaHeader); }



}  // namespace uTensor
