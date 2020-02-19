#include "SimpleErrorHandler.hpp"

namespace uTensor {

SimpleErrorHandler::SimpleErrorHandler(size_t max_num_events) : max_num_events(max_num_events) {}

void SimpleErrorHandler::uThrow(Error* err)  {
  if(_onError)
    return _onError(err);
  else {
    while(true) {}
  }

}
void SimpleErrorHandler::notify(const Event& evt)  {
  if(_eventQ.size() >= max_num_events){
    _eventQ.pop_front();
  }
  _eventQ.push_back(evt);
}

void SimpleErrorHandler::set_onError(std::function<void(Error*)> onError) {
  _onError = onError;
}
std::deque<Event>::iterator SimpleErrorHandler::begin(){
  return _eventQ.begin();
}
std::deque<Event>::iterator SimpleErrorHandler::end(){
  return _eventQ.end();
} 

}
