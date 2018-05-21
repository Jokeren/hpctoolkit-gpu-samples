#ifndef _CFG_PARSER_H_
#define _CFG_PARSER_H_

#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include "inst.h"
#include "cfg.h"
#include "graph.h"

class CFGParser {
 public:
  CFGParser() : _block_parent(0) {}
  void parse(const Graph &graph, std::vector<Function *> &functions);

 private:
  void parse_inst_strings(const std::string &label, std::deque<std::string> &inst_strings);

  size_t find_block_parent(size_t node);

  void unite_blocks(size_t l, size_t r);

 private:
  std::vector<size_t> _block_parent;
};

#endif
