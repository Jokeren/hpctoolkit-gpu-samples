#include "cfg_analyzer.h"
#include <string>

std::vector<Call *> CFGAnalyzer::extract_calls() {
  std::vector<Call *> calls;
  for (auto function : _functions) {
    for (auto block : function->blocks) {
      for (auto inst : block->insts) {
        if (inst->opcode.find("CALL") != std::string::npos || // sm_70
          inst->opcode.find("CAL") != std::string::npos) { // sm_60
          std::string &operand = inst->operands[0];
          std::string callee = operand.substr(2, operand.size() - 4);
          Function *callee_function;
          for (auto ff : _functions) {
            if (ff->name == callee) {
              callee_function = ff;
              break;
            }
          }
          calls.push_back(new Call(inst, block, function, callee_function));
        }
      }
    }
  }
  return calls;
}


bool CFGAnalyzer::WMZC_tag_head(Block* b, Block* h) {
  if (b == h || h == NULL) return false;
  Block *cur1, *cur2;
  cur1 = b; cur2 = h;
  while (_block_header[cur1] != NULL) {
    Block* ih = _block_header[cur1];
    if (ih == cur2) return true;
    if (_DFS_pos[ih] < _DFS_pos[cur2]) { // Can we guarantee both are not 0?
      _block_header[cur1] = cur2;
      cur1 = cur2;
      cur2 = ih;
    } else cur1 = ih;
  }
  _block_header[cur1] = cur2;
  return true;
}


Block *CFGAnalyzer::WMZC_DFS(Function *func, Block* b0, size_t pos) {
  _visited_blocks.insert(b0);
  _DFS_pos[b0] = pos;
  // The final loop nesting structure depends on
  // the order of DFS. To guarantee that we get the 
  // same loop nesting structure for an individual binary 
  // in all executions, we sort the target blocks using
  // the start adress.
  std::vector<Target *> visited_order(b0->targets.begin(), b0->targets.end());
  for (auto t : visited_order) {
    Block *b = t->block;
    if (_visited_blocks.find(b) == _visited_blocks.end()) {
      // case A, new
      Block *nh = WMZC_DFS(func, b, pos + 1);
      WMZC_tag_head(b0, nh);
    } else {
      if (_DFS_pos[b] > 0) {
        // case B
        if (_block_loop[b] == NULL)
          _block_loop[b] = new Loop(func);
        if (WMZC_tag_head(b0, b)) {
          _block_loop[b]->entries.push_back(new LoopEntry(b));
        }
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
          _block_loop[h]->entries.push_back(new LoopEntry(b));
          while (_block_header[h] != NULL) {
            h = _block_header[h];
            if (_DFS_pos[h] > 0) {
              WMZC_tag_head(b0, h);
              break;
            }  
            _block_loop[h]->entries.push_back(new LoopEntry(b));
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

  std::unordered_set<Block *> block_erase;
  for (auto child_block : cur_loop->blocks) {
    auto child_loop = _block_loop[child_block];
    create_loop_hierarchy(child_block);
    if (child_loop != NULL) {
      block_erase.insert(child_block);
      cur_loop->child_loops.insert(child_loop);
      for (auto cb : child_loop->blocks) {
        cur_loop->child_blocks.insert(cb);
      }
      for (auto ccb : child_loop->child_blocks) {
        cur_loop->child_blocks.insert(ccb);
      }
    }
  }

  cur_loop->blocks.insert(cur);
  for (auto block : block_erase) {
    cur_loop->blocks.erase(block);
    cur_loop->child_blocks.insert(block);
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

    // Self contained loops
    for (auto block : function->blocks) {
      for (auto target : block->targets) {
        if (target->block == block) {
          _block_header[block] = block;
          _block_loop[block]->entries.push_back(new LoopEntry(block));
          break;
        }
      }
    }

    for (auto block : function->blocks) {
      if (_block_header[block] != NULL) {
        _block_loop[_block_header[block]]->blocks.insert(block);
      }
      //std::cout << "Block header: " << std::endl;
      //std::cout << _block_header[block]->name << std::endl;
      //std::cout << "Blocks: " << std::endl;
      //std::cout << block->name << std::endl;
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

    // Enumerate entry back edge blocks and insts
    for (auto lit : loop_set) {
      std::vector<LoopEntry *> loop_entries;
      Loop *loop = lit;
      // Fill back_edge_block
      for (auto entry : loop->entries) {
        for (auto block : loop->blocks) {
          for (auto target : block->targets) {
            if (target->block == entry->entry_block) {
              loop_entries.push_back(new LoopEntry(entry->entry_block, block, target->inst));
            }
          }
        }
        for (auto block : loop->child_blocks) {
          for (auto target : block->targets) {
            if (target->block == entry->entry_block) {
              loop_entries.push_back(new LoopEntry(entry->entry_block, block, target->inst));
            }
          }
        }
      }
      loop->entries = loop_entries;
    }

    for (auto lit : loop_set) {
      loops.push_back(lit);
    }
  }

  return loops;
}
