#include "cfg_parser.h"
#include <cctype>

void CFGParser::parse_inst_strings(
  const std::string &label,
  std::deque<std::string> &inst_strings) {
  std::regex e("\\\\l([|]*)");
  std::istringstream ss(std::regex_replace(label, e, "\n"));
  std::string s;
  while (std::getline(ss, s)) {
    inst_strings.push_back(s);
  }
  while (inst_strings.size() > 0) {
    if (isdigit(inst_strings.front()[0]) || inst_strings.front()[0] == '<') {
      break;
    }
    inst_strings.pop_front();
  }
  inst_strings.pop_back();
}


size_t CFGParser::find_block_parent(size_t node) {
  size_t parent = _block_parent[node];
  size_t graph_size = _block_parent.size();
  if (parent == graph_size) {
    return _block_parent[node] = node;
  } else if (parent == node) {
    return node;
  } else {
    return _block_parent[node] = find_block_parent(parent);
  }
}


void CFGParser::unite_blocks(size_t l, size_t r) {
  _block_parent[l] = find_block_parent(r);
}


static bool compare_block_ptr(Block *l, Block *r) {
  return *l < *r;
}


static bool compare_target_ptr(Target *l, Target *r) {
  return *l < *r;
}


void CFGParser::parse(const Graph &graph, std::vector<Function *> &functions) {
  std::unordered_map<size_t, Block *> block_map;
  std::vector<Block *> blocks;
  size_t graph_size = graph.vertices.size();
  _block_parent.resize(graph_size);
  for (size_t i = 0; i < graph_size; ++i) {
    _block_parent[i] = graph_size;
  }

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
        source_block->targets.push_back(new Target(inst, target_block));
      }
    }
    // Some edge may not have port information
    if (source_block->targets.size() == 0) {
      source_block->targets.push_back(new Target(source_block->insts.back(), target_block));
    }
  }

  //for (auto block : blocks) {
  //  std::cout << "From: " << std::endl;
  //  std::cout << block->name << std::endl;
  //  std::cout << "Target: " << std::endl;
  //  for (auto target : block->targets) {
  //    std::cout << target->block->name << std::endl;
  //  }
  //}

  // Build functions
  size_t function_id = 0;
  for (auto block : blocks) {
    // Sort block targets according to inst offset
    std::sort(block->targets.begin(), block->targets.end(), compare_target_ptr);
    if (find_block_parent(block->id) == block->id) {
      // Filter out self contained useless loops. A normal function will not start with "."
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
      std::sort(function->blocks.begin(), function->blocks.end(), compare_block_ptr);
      functions.push_back(function);
    }
  }
}
