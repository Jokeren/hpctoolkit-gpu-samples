#ifndef _CFG_H_
#define _CFG_H_

#include <sstream>
#include <vector>
#include <string>
#include "inst.h"


struct Block {
  std::vector<Inst *> insts;
  size_t id;
  std::string name;

  Block(size_t id, std::string &name) : id(id), name(name) {}

  bool operator<(const Block &other) {
    if (this->insts.size() == 0) {
      return true;
    } else if (other.insts.size() == 0) {
      return false;
    } else {
      return this->insts[0]->offset < other.insts[0]->offset;
    }
  }
};


struct Function {
  std::vector<Block *> blocks;
  size_t id;
  std::string name;

  Function(size_t id, std::string &name) : id(id), name(name) {}
};


struct Loop {
  Block *entry_block; 
  Block *back_edge_block;
  Function *function;

  Loop(Block *entry_block, Block *back_edge_block, Function *function) :
    entry_block(entry_block), back_edge_block(back_edge_block), function(function) {}
};


struct Call {
  Block *caller_block; 
  Function *caller_function;
  Function *callee_function;

  Call(Block *caller_block, Function *caller_function, Function *callee_function) :
    caller_block(caller_block), caller_function(caller_function), callee_function(callee_function) {}
};

#endif
