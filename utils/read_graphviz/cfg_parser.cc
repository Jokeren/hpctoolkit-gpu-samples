#include "cfg_parser.h"

void CFGParser::parse_inst_strings(
  const std::string &label,
  std::deque<std::string> &inst_strings) {
  std::regex e("\\\\l([|]*)");
  std::istringstream ss(std::regex_replace(label, e, "\n"));
  std::string s;
  while (std::getline(ss, s)) {
    inst_strings.push_back(s);
  }
  inst_strings.pop_front();
  inst_strings.pop_back();
}


size_t CFGParser::find_block_parent(size_t node) {
  auto iter = _block_parent.find(node);
  if (iter == _block_parent.end()) {
    return _block_parent[node] = node;
  } else if (iter->second == node) {
    return node;
  } else {
    return _block_parent[node] = find_block_parent(iter->second);
  }
}


void CFGParser::unite_blocks(size_t l, size_t r) {
  _block_parent[l] = find_block_parent(r);
}


void CFGParser::parse(const Graph &graph, std::vector<Function *> &functions) {
  std::unordered_map<size_t, Block *> block_map;
  std::vector<Block *> blocks;
  _block_parent.clear();

  // Parse every vertex to build blocks
  for (auto vertex : graph.vertices) {
    Block *block = new Block(vertex->id, vertex->name);

    std::deque<std::string> inst_strings;
    parse_inst_strings(vertex->label, inst_strings);
    for (auto inst_string : inst_strings) {
      block->insts.push_back(new Inst(inst_string));
    }

    blocks.push_back(block);
    block_map[block->id] = block;
  }

  // Parse every edge to build block relations
  for (auto edge : graph.edges) {
    // Find toppest block
    unite_blocks(edge->target_id, edge->source_id);
    Block *target_block = block_map[edge->target_id];
    Block *source_block = block_map[edge->source_id];
    // Link blocks
    Inst *target_inst;
    for (auto inst : source_block->insts) {
      if (inst->port == edge->source_port[0]) {
        target_inst = inst;
        break;
      }
    }
    source_block->targets.push_back(new Target(target_inst, target_block));
  }

  // Build functions
  size_t function_id = 0;
  for (auto block : blocks) {
    // Sort block targets according to inst offset
    std::sort(block->targets.begin(), block->targets.end());
    if (find_block_parent(block->id) == block->id) {
      // Filter out self contained useless loops. A normal function will start with "."
      if (block_map[block->id]->name[0] == '.') {
        continue;
      }
      Function *function = new Function(function_id, block_map[block->id]->name);
      ++function_id;
      for (auto bb : blocks) {
        if (find_block_parent(bb->id) == block->id) {
          function->blocks.push_back(bb);
        }
      }
      std::sort(function->blocks.begin(), function->blocks.end());
      functions.push_back(function);
    }
  }
}
