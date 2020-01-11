#ifndef UTENSOR_ERROR_H
#define UTENSOR_ERROR_H

namespace uTensor {

struct Error {};

// Simplest possible error handler, Users can roll their own
class ErrorHandler {
    public:
        virtual void uThrow(Error* err);
};

// Default errors
struct InvalidReshapeError : public Error {};
struct InvalidResizeError : public Error {};
struct InvalidMemAccessError : public Error {};

};
#endif
