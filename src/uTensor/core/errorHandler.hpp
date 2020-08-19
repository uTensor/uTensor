#ifndef UTENSOR_ERROR_H
#define UTENSOR_ERROR_H

/** Error Handling and Event primatives
 * Rather than relying on RTTI for complete compile (fairly expensive for tiny
 * devices) we opt to hash the event names at compile time, and store them at
 * runtime. The ROM/RAM overheads on these are fixed integral widths and
 * configurable via the event_handle_type.
 */

// check if the follow modificiation affect other builds
//#include <cstdint>
#include <stdint.h>

namespace uTensor {

using event_handle_type = uint16_t;

// Compile time hash borrowed from
// https://gist.github.com/underscorediscovery/81308642d0325fd386237cfa3b44785c
#define uT_STRINGIFY(x) #x
// FNV1a c++11 constexpr compile time hash functions, 32 and 64 bit
// str should be a null terminated string literal, value should be left out
// e.g hash_32_fnv1a_const("example")
// code license: public domain or equivalent
// post: https://notes.underscorediscovery.com/constexpr-fnv1a/

constexpr uint32_t val_32_const = 0x811c9dc5;
constexpr uint32_t prime_32_const = 0x1000193;
constexpr uint64_t val_64_const = 0xcbf29ce484222325;
constexpr uint64_t prime_64_const = 0x100000001b3;

inline constexpr uint32_t hash_32_fnv1a_const(
    const char* const str, const uint32_t value = val_32_const) noexcept {
  return (str[0] == '\0')
             ? value
             : hash_32_fnv1a_const(&str[1],
                                   (value ^ uint32_t(str[0])) * prime_32_const);
}
inline constexpr event_handle_type u32toEventType(const uint32_t val) noexcept {
  return (event_handle_type)val;
}

struct Event {
  event_handle_type event_id;
};

struct Error : public Event {
  Error(event_handle_type t);
};

// Simplest possible error handler, Users can roll their own
class ErrorHandler {
 public:
  virtual void uThrow(Error* err);
  virtual void notify(const Event& evt);
};

// UID gets evaluated at compile time and can be looked up at runtime :D
// Let's us fake RTTI for the bits we care about
#define DECLARE_EVENT(EVT)                                   \
  struct EVT : public Event {                                \
    static constexpr uint16_t uid =                          \
        u32toEventType(hash_32_fnv1a_const(uT_STRINGIFY(EVT))); \
    EVT();                                                   \
  }
#define DEFINE_EVENT(EVT) \
  EVT::EVT() : Event{uid} {}
#define DECLARE_ERROR(EVT)                                   \
  struct EVT : public Error {                                \
    static constexpr uint16_t uid =                          \
        u32toEventType(hash_32_fnv1a_const(uT_STRINGIFY(EVT))); \
    EVT();                                                   \
  }
#define DEFINE_ERROR(EVT) \
  EVT::EVT() : Error{uid} {}

bool operator==(const Event& a, const Event& b);

// Default errors
DECLARE_ERROR(InvalidReshapeError);
DECLARE_ERROR(InvalidResizeError);
DECLARE_ERROR(InvalidMemAccessError);
DECLARE_ERROR(OutOfMemError);
DECLARE_ERROR(OutOfMemBoundsError);
DECLARE_ERROR(InvalidOptimizableTensorError);
DECLARE_ERROR(InvalidTensorError);
DECLARE_ERROR(InvalidTensorInputError);
DECLARE_ERROR(InvalidTensorOutputError);
DECLARE_ERROR(InvalidTensorDimensionsError);
DECLARE_ERROR(InvalidTensorDataTypeError);

};  // namespace uTensor
#endif
