#ifndef _CFG_ANALYZER_H_
#define _CFG_ANALYZER_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "cfg.h"

class CFGAnalyzer {
 public:
  CFGAnalyzer(const std::vector<Function *> &functions) : _functions(functions) {}
  std::vector<Call *> extract_calls();
  std::vector<Loop *> extract_loops();

 private:
  Block *WMZC_DFS(Function *func, Block *b0, size_t pos);
  void WMZC_tag_head(Block *b, Block *h);
  void create_loop_hierarchy(Block *cur);

 private:
  std::vector<Function *> _functions;
  std::unordered_map<Block *, Block *> _block_header;
  std::unordered_map<Block *, size_t> _DFS_pos;
  std::unordered_map<Block *, Loop *> _block_loop;
  std::unordered_set<Block *> _visited_blocks;
};

#endif
