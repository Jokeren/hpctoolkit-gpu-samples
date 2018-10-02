#include <random>
#include <iostream>
#include <set>
#include "splay.h"

void simple_test() {
  SplayTree t;
  t.insert(5);
  t.insert(1);
  t.insert(3);
  t.insert(6);
  t.insert(2);
  t.insert(7);
  t.remove(1);
  t.remove(5);
  t.traverse();
}

void random_test() {
  std::set<int> s;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist(0, 100);

  SplayTree t;
  for (size_t i = 0; i < 1000; ++i) {
    int num = dist(mt);
    t.insert(num);
    s.insert(num);
  }
  for (size_t i = 0; i < 200; ++i) {
    int num = dist(mt);
    t.remove(num);  
    s.erase(num);
  }
  for (auto num : s) {
    if (!t.lookup(num)) {
      std::cout << "Fault: " << num << std::endl;
    } else {
      std::cout << "Correct: " << num << std::endl;
    }
  }
}

int main() {
  simple_test();
  random_test();
  return 0;
}
