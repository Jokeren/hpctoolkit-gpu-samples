#ifndef _CUPTI_RECORD_H_
#define _CUPTI_RECORD_H_

#include "cupti_stack.h"

template<typename T>
class CuptiRecord {
 public:
  CuptiRecord() {}

  CuptiStack<T> worker_notification_stack;
  CuptiStack<T> worker_free_notification_stack;
  CuptiStack<T> cupti_notification_stack;
  CuptiStack<T> cupti_free_notification_stack;

  size_t recycled = 0;

  void worker_notification_apply(int id) {
    CuptiStackItem<T> *node = worker_free_notification_stack.pop();
    if (node == NULL) {
      worker_free_notification_stack.set_top(cupti_free_notification_stack.steal());
      node = worker_free_notification_stack.pop();
    }
    if (node == NULL) {
      node = new CuptiStackItem<T>(id);
    } else {
      ++recycled;
      node->set_value(id);
    } 
    worker_notification_stack.push(node);
  }

  size_t cupti_notification_apply(CuptiRecord &record) {
    CuptiStack<T> &worker_notification_stack = record.worker_notification_stack;
    CuptiStack<T> &cupti_notification_stack = record.cupti_notification_stack;
    cupti_notification_stack.set_top(worker_notification_stack.steal());
    size_t consume = 0;
    if (cupti_notification_stack.top() != NULL) {
      CuptiStack<T> &cupti_free_notification_stack = record.cupti_free_notification_stack;
      CuptiStackItem<T> *node = NULL;
      while (node = cupti_notification_stack.pop()) {
        ++consume;
        cupti_free_notification_stack.push(node);
      }
    }
    return consume;
  }
};

#endif
