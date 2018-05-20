#ifndef _CFG_H_
#define _CFG_H_

#include <sstream>
#include <vector>
#include <string>
#include "inst.h"

struct Block;

struct Target {
  Inst *inst;
  Block *block;

  Target(Inst *inst, Block *block) : inst(inst), block(block) {}

  bool operator<(const Target &other) {
    return this->inst->offset < other.inst->offset;
  }
};


struct Block {
  std::vector<Inst *> insts;
  std::vector<Target *> targets;
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

  Function(size_t id, const std::string &name) : id(id), name(name) {}
};


struct LoopEntry {
  Block *entry_block; 
  Block *back_edge_block;
  Inst *back_edge_inst;

  LoopEntry(Block *entry_block) : entry_block(entry_block) {}

  LoopEntry(Block *entry_block, Block *back_edge_block, Inst *back_edge_inst) :
    entry_block(entry_block), back_edge_block(back_edge_block), back_edge_inst(back_edge_inst) {}
};


struct Loop {
  std::vector<LoopEntry *> entries;
  std::vector<Loop *> child_loops;
  std::vector<Block *> blocks;
  Function *function;

  Loop(Function *function) : function(function) {}
};


struct Call {
  Block *caller_block; 
  Function *caller_function;
  Function *callee_function;

  Call(Block *caller_block, Function *caller_function, Function *callee_function) :
    caller_block(caller_block), caller_function(caller_function), callee_function(callee_function) {}
};

#endif
