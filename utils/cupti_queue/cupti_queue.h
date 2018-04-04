#ifndef _CUPTI_QUEUE_H_
#define _CUPTI_QUEUE_H_

#include <atomic>
#include <iterator>

template<typename T>
class CuptiQueueItem {
private:
  typedef CuptiQueueItem<T> item_type;

public:
  CuptiQueueItem(T __value) : _next(NULL), _value(__value) {
  };

  ~CuptiQueueItem() { 
  };

  void setNext(item_type *__next) { 
    _next.store(__next); 
  };

  item_type *next() { 
    item_type *succ = _next.load();
    return succ;
  };

  T value() { 
    return _value; 
  };

private:
  std::atomic<item_type *> _next;
  T _value;
};


template<typename T>
class CuptiQueueIterator {
private:
  typedef CuptiQueueItem<T> item_type;
  typedef CuptiQueueIterator<T> iterator;

public:
  typedef T value_type;
  typedef T * pointer;
  typedef T & reference;
  typedef std::ptrdiff_t difference_type;
  typedef std::forward_iterator_tag iterator_category;

public:
  CuptiQueueIterator(item_type *_item) : item(_item) {};

  T operator *() { return item->value(); };

  bool operator != (iterator i) { return item != i.item; };

  iterator operator ++() { 
    if (item) item = item->next(); 
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
class CuptiQueue {
public:
  typedef CuptiQueueIterator<T> iterator;
  typedef CuptiQueueItem<T> item_type; 

public:
  CuptiQueue(item_type *head = 0, item_type *tail = 0) : _head(head), _tail(tail) {};

public: 
  void insert(T value) {
    item_type *entry = new item_type(value);
    item_type *last = _tail.load();
    entry->setNext(last); 
    if (!_tail.compare_exchange_strong(last, entry)) {  // fail because of splice
      entry->setNext(NULL); 
      _tail.store(entry);
    }
  }

  // steal the linked list from q and insert it at the front of this queue
  void splice(CuptiQueue<T> &q) {
    item_type *q_last = q.steal();
    item_type *first = _head.load();
    item_type *last = _tail.load();
    if (q_last) {  // non-empty q
      item_type *prev = NULL;  // linked-list reverse
      item_type *node = q_last;
      while (node) {
        item_type *next = node->next();
        node->setNext(prev);
        prev = node;
        node = next;
      }
      if (first) { // this q is non-empty
        last->setNext(prev);
      } else { // this q is empty
        _head.store(prev);
      }
      _tail.store(q_last);
    }
  };

  // inspect the head of the queue
  item_type *peek() { return _head.load(); };

  // inspect the head of the queue
  item_type *rear() { return _tail.load(); };

  // grab the contents of the queue for your own private use
  item_type *steal() { return _tail.exchange(NULL); };

  // inspect the head of the queue
  void setPeek(item_type *node) { return _head.store(node); };

  iterator begin() { return iterator(_head.load()); };

  iterator end() { return iterator(NULL); };

  item_type *pop() {
    item_type *first = _head.load();
    if (first) {
      item_type *succ = first->next(); 
      _head.store(succ);
      return first;
    } else {
      return NULL;
    }
  };

  ~CuptiQueue() { clear(); };

private:
  void clear() { 
    item_type *first = NULL;
    while((first = pop())) { 
      delete first;
    }
  };

private:
  std::atomic<item_type *> _head, _tail;
};

#endif

