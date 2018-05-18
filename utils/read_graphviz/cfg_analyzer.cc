#include "cfg_analyzer.h"
#include <string>

std::vector<Call *> CFGAnalyzer::extract_calls() {
  std::vector<Call *> calls;
  for (auto function : _functions) {
    for (auto block : function->blocks) {
      size_t num_insts = block->insts.size();
      if (num_insts > 0) {
        Inst *inst = block->insts[num_insts - 1];
        if (inst->opcode.find("CALL") != std::string::npos) {
          std::string &operand = inst->operands[0];
          std::string callee = operand.substr(2, operand.size() - 4);
          Function *callee_function;
          for (auto ff : _functions) {
            if (ff->name == callee) {
              callee_function = ff;
              break;
            }
          }
          calls.push_back(new Call(block, function, callee_function));
        }
      }
    }
  }
  return calls;
}


void CFGAnalyzer::WMZC_tag_head(Block* b, Block* h) {
  if (b == h || h == NULL) return;
  Block *cur1, *cur2;
  cur1 = b; cur2 = h;
  while (_block_header[cur1] != NULL) {
    Block* ih = _block_header[cur1];
    if (ih == cur2) return;
    if (_DFS_pos[ih] < _DFS_pos[cur2]) { // Can we guarantee both are not 0?
      _block_header[cur1] = cur2;
      cur1 = cur2;
      cur2 = ih;
    } else cur1 = ih;
  }
  _block_header[cur1] = cur2;
}


Block *CFGAnalyzer::WMZC_DFS(Function *func, Block* b0, size_t pos) {
  _visited_blocks.insert(b0);
  _DFS_pos[b0] = pos;
  // The final loop nesting structure depends on
  // the order of DFS. To guarantee that we get the 
  // same loop nesting structure for an individual binary 
  // in all executions, we sort the target blocks using
  // the start adress.
  std::vector<Block *> visited_order(b0->targets.begin(), b0->targets.end());
  sort(visited_order.begin(), visited_order.end());
  for (auto b : visited_order) {
    if (_visited_blocks.find(b) == _visited_blocks.end()) {
      // case A, new
      Block *nh = WMZC_DFS(func, b, pos + 1);
      WMZC_tag_head(b0, nh);
    } else {
      if (_DFS_pos[b] > 0) {
        // case B
        if (_block_loop[b] == NULL)
          _block_loop[b] = new Loop(func);
        WMZC_tag_head(b0, b);
        _block_loop[b]->entries.push_back(new LoopEntry(b, b0));
      } else if (_block_header[b] == NULL) {
        // case C, do nothing
      } else {
        Block *h = _block_header[b];
        if (_DFS_pos[h] > 0) {
          // case D
          WMZC_tag_head(b0, h);
        } else {
          // case E
          // Mark b and (b0,b) as re-entry
          _block_loop[h]->entries.push_back(new LoopEntry(b, NULL));
          while (_block_header[h] != NULL) {
            h = _block_header[h];
            if (_DFS_pos[h] > 0) {
              WMZC_tag_head(b0, h);
              break;
            }  
            _block_loop[h]->entries.push_back(new LoopEntry(b, NULL));
          }
        }
      }
    }
  }
  _DFS_pos[b0] = 0;
  return _block_header[b0];
}


// Recursively build the basic blocks in a loop
// and the contained loops in a loop
void CFGAnalyzer::create_loop_hierarchy(Block *cur) {
  auto cur_loop = _block_loop[cur];
  if(cur_loop == NULL) 
    return;

  std::unordered_set<Loop *> loop_filter;
  for (auto child_block : cur_loop->blocks) {
    auto child_loop = _block_loop[child_block];
    if (child_loop != NULL) {
      if (loop_filter.find(child_loop) == loop_filter.end()) {
        loop_filter.insert(child_loop);
        cur_loop->child_loops.push_back(child_loop);
      }
    }
    create_loop_hierarchy(child_block);
  }
}


std::vector<Loop *> CFGAnalyzer::extract_loops() {
  std::vector<Loop *> loops;

  for (auto function : _functions) {
    // Init local variables
    _DFS_pos.clear();
    _visited_blocks.clear();
    _block_loop.clear();
    _block_header.clear();
    std::unordered_set<Loop *> loop_set;

    // Apply loop finding algorithm
    WMZC_DFS(function, function->blocks[0], 1);

    for (auto block : function->blocks) {
      if (_block_header[block] == NULL)
        continue;
      _block_loop[_block_header[block]]->blocks.push_back(block);
    }

    for (auto block : function->blocks) {
      // Find loop header
      if (_block_header[block] == NULL) {
        create_loop_hierarchy(block);
      }
    }

    for (auto block : function->blocks) {
      if (_block_loop[block] != NULL) {
        loop_set.insert(_block_loop[block]);
      }
    }

    for (auto lit : loop_set) {
      Loop *loop = lit;
      for (auto entry : loop->entries) {
        if (entry->back_edge_block == NULL) {
          for (auto block : loop->blocks) {
            if (std::find(block->targets.begin(), block->targets.end(), entry->entry_block) != block->targets.end()) {
              entry->back_edge_block = block;
            }
          }
        }
      }
    }

    for (auto lit : loop_set) {
      loops.push_back(lit);
    }
  }

  return loops;
}
