#ifndef _CUPTI_STACK_H_
#define _CUPTI_STACK_H_

#include <atomic>
#include <iterator>

template<typename T>
class CuptiStackItem {
 public:
  typedef CuptiStackItem<T> item_type;

 public:
  item_type *_next;

 public:
  CuptiStackItem(T value) : _next(NULL), _value(value) {
  };

  ~CuptiStackItem() { 
  };

  T value() { 
    return _value; 
  };

  void set_value(T &value) {
    _value = value;
  }

 private:
  T _value;
};


template<typename T>
class CuptiStackIterator {
 private:
  typedef CuptiStackItem<T> item_type;
  typedef CuptiStackIterator<T> iterator;

 public:
  typedef T value_type;
  typedef T * pointer;
  typedef T & reference;
  typedef std::ptrdiff_t difference_type;
  typedef std::forward_iterator_tag iterator_category;

 public:
  CuptiStackIterator(item_type *_item) : item(_item) {};

  T operator *() { return item->value(); };

  bool operator != (iterator i) { return item != i.item; };

  iterator operator ++() { 
    if (item) item = item->_next;
    return *this; 
  };

  iterator operator ++(int) { 
    iterator clone(*this);
    ++(*this);
    return clone;
  };

 private:
  item_type *item;
};


template<typename T>
class CuptiStack {
 public:
  typedef CuptiStackIterator<T> iterator;
  typedef CuptiStackItem<T> item_type; 

 public:
  CuptiStack(item_type *head = 0) : _head(head) {};

 public: 
  void push(item_type *entry) {
    item_type *last = top();
    entry->_next = last; 
    while (!_head.compare_exchange_strong(entry->_next, entry));
  }

  item_type *pop() {
    item_type *first = top();
    if (first) {
      item_type *succ = first->_next;
      while (first && !_head.compare_exchange_strong(first, succ)) {
        succ = first->_next;
      }
      return first;
    } else {
      return NULL;
    }
  };

  void splice(CuptiStack<T> &other) {
    item_type *first = other.steal();
    if (first) {
      item_type *last = first;
      item_type *next = first->_next;
      while (next != NULL) {
        last = next;
        next = next->_next;
      }
      last->_next = top();
      set_top(first);
    }
  }

  size_t size() {
    size_t curr_size = 0;
    item_type *first = top();
    if (first) {
      ++curr_size;
      item_type *next = first->_next;
      while (next != NULL) {
        ++curr_size;
        next = next->_next;
      }
    }
    return curr_size;
  }

  // inspect the head of the stack
  item_type *top() { return _head.load(); };

  // grab the contents of the stack for your own private use
  item_type *steal() { return _head.exchange(NULL); };

  // change the head of the stack
  void set_top(item_type *node) { return _head.store(node); };

  iterator begin() { return iterator(_head.load()); };

  iterator end() { return iterator(NULL); };

  ~CuptiStack() { clear(); };

 private:
  void clear() { 
    item_type *first = NULL;
    while((first = pop())) { 
      delete first;
    }
  };

 private:
  std::atomic<item_type *> _head;
};

#endif
