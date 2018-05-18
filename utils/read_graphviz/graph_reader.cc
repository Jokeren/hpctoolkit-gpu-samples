#include "graph_reader.h"
#include "inst.h"

std::vector<Function *> GraphReader::construct_cfg() {
  // Declare properties
  boost::dynamic_properties dp;

  dp.property("node_id", get(&Vertex::name, _g));
  dp.property("fontname", get(&Vertex::font_name, _g));
  dp.property("fontsize", get(&Vertex::font_size, _g));
  dp.property("shape", get(&Vertex::shape, _g));
  dp.property("label", get(&Vertex::label, _g));
  dp.property("style", get(&Edge::style, _g));

  // Read dot graph
  std::ifstream file(_file_name);
  std::stringstream dotfile;

  dotfile << file.rdbuf();
  file.close();
  boost::read_graphviz(dotfile, _g, dp);

  // Construct CFG
  _block_parent.clear();
  std::vector<Block *> blocks = construct_blocks();
  std::vector<Function *> functions = construct_functions(blocks);
  return functions;
}


size_t GraphReader::find_block_parent(size_t node) {
  auto iter = _block_parent.find(node);
  if (iter == _block_parent.end()) {
    return _block_parent[node] = node;
  } else if (iter->second == node) {
    return node;
  } else {
    return _block_parent[node] = find_block_parent(iter->second);
  }
}


void GraphReader::unite_blocks(size_t l, size_t r) {
  _block_parent[l] = find_block_parent(r);
}


std::vector<Block *> GraphReader::construct_blocks() {
  //std::cout << std::endl << "Every vertex pos:" << std::endl;
  std::unordered_map<size_t, Block *> block_map;
  std::vector<Block *> blocks;

  std::regex e("\\\\l([|]*)");
  BGL_FORALL_VERTICES(v, _g, Graph) {
    //std::cout << v << ": " << _g[v].name << std::endl;
    Block *block = new Block(v, _g[v].name);

    std::string label = _g[v].label;
    std::istringstream ss(std::regex_replace(label, e, "\n"));
    std::deque<std::string> inst_strings;
    std::string s;
    while (std::getline(ss, s)) {
      inst_strings.push_back(s);
    }
    inst_strings.pop_front();
    inst_strings.pop_back();

    for (auto inst_string : inst_strings) {
      Inst *inst = new Inst(inst_string);
      block->insts.push_back(inst);
    }
    blocks.push_back(block);
    block_map[v] = block;
  }

  //std::cout << std::endl << "Every edge pos:" << std::endl;
  BGL_FORALL_EDGES(e, _g, Graph) {
    //std::cout << _g[source(e,_g)].name << " -> " << _g[target(e,_g)].name << ": " << _g[e].style << std::endl;
    // Adjusted union find operation
    unite_blocks(boost::target(e, _g), boost::source(e, _g));
    Block *block1 = block_map[boost::source(e, _g)];
    Block *block2 = block_map[boost::target(e, _g)];
    block1->targets.push_back(block2);
  }

  return blocks;
}


std::vector<Function *> GraphReader::construct_functions(const std::vector<Block *> &blocks) {
  // Find every block's source node
  // Build functions
  std::vector<Function *> functions;
  size_t function_id = 0;
  for (auto block : blocks) {
    if (find_block_parent(block->id) == block->id) {
      Function *function = new Function(function_id, _g[block->id].name);
      //std::cout << function->name << std::endl;
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
  return functions;
}
