#include <cstdio>

#include "uTensor/core/context.hpp"
#include "errorHandler.hpp"

using namespace std;
using namespace uTensor;

// declare and define your own event type
// we will try to catch MyEvent but not EventDontCatch
DECLARE_EVENT(MyEvent);
DEFINE_EVENT(MyEvent);
DECLARE_EVENT(EventDontCatch);
DEFINE_EVENT(EventDontCatch);

// declare and define your own error type
DECLARE_ERROR(MyError);
DEFINE_ERROR(MyError);

class MyEventHandler : public ErrorHandler {
 public:
  // uThrow: normally invoked when a error occurs during op evaluation
  // you can override this method and do what ever you want with the error.
  // In this tutorial, we're going to implement a spin-wait handler so we
  // can debug the error easily in a debugger such as lldb or gdb.
  virtual void uThrow(Error* err) override {
    if (err->event_id == MyError::uid) {
      printf("MyError thrown. Spinning...\n");
      while (true) {
      }
    }
  }

  // notify: normally invoked when a framework event occurs. For example, in
  // the arenaAllocator, events such as the allocator is created will notified
  // to the handler. You can do check/logging with it.
  // Each event is identified by a static uid.
  virtual void notify(const Event& evt) override {
    if (evt.event_id == MyEvent::uid) {
      printf("MyEvent detected\n");
    } else {
      printf("Unknown Event detected\n");
    }
  }
};

int main(int argc, const char** argv) {
  MyEventHandler handler;
  Context* ptrDefaultContext = Context::get_default_context();
  // setup handler
  ptrDefaultContext->set_ErrorHandler(&handler);
  // this event won't be handled
  ptrDefaultContext->notifyEvent(EventDontCatch());
  // but this event will be handled
  ptrDefaultContext->notifyEvent(MyEvent());
  // this error will not be handled in our handler
  ptrDefaultContext->throwError(new InvalidTensorError);
  // but this error will be handled and cause the process to spin
  ptrDefaultContext->throwError(new MyError);
  return 0;
}