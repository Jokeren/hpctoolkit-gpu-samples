#include "graph_reader.h"
#include "inst.h"

void GraphReader::construct_cfg() {
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
  construct_blocks();
  construct_functions();
}


void GraphReader::construct_blocks() {
  std::regex e("\\\\l([|]*)");
  //std::cout << std::endl << "Every vertex pos:" << std::endl;
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
    _blocks.push_back(block);
  }
}


size_t GraphReader::find_block_parents(size_t node) {
  auto iter = _block_parents.find(node);
  if (iter == _block_parents.end()) {
    return _block_parents[node] = node;
  } else if (iter->second == node) {
    return node;
  } else {
    return _block_parents[node] = find_block_parents(iter->second);
  }
}


void GraphReader::unite_blocks(size_t l, size_t r) {
  _block_parents[l] = find_block_parents(r);
}


void GraphReader::construct_functions() {
  //std::cout << std::endl << "Every edge pos:" << std::endl;
  BGL_FORALL_EDGES(e, _g, Graph) {
    //std::cout << _g[source(e,_g)].name << " -> " << _g[target(e,_g)].name << ": " << _g[e].style << std::endl;
    // Adjusted union find operation
    unite_blocks(boost::target(e, _g), boost::source(e, _g));
  }

  // Find every block's source node
  // Build functions
  size_t function_id = 0;
  for (auto block : _blocks) {
    if (find_block_parents(block->id) == block->id) {
      Function *function = new Function(function_id, _g[block->id].name);
      //std::cout << function->name << std::endl;
      ++function_id;
      for (auto bb : _blocks) {
        if (find_block_parents(bb->id) == block->id) {
          function->blocks.push_back(bb);
        }
      }
      std::sort(function->blocks.begin(), function->blocks.end());
      _functions.push_back(function);
    }
  }
}


std::vector<Call *> GraphReader::analyze_calls() {
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


std::vector<Loop *> GraphReader::analyze_loops() {
  std::vector<Loop *> loops;
  return loops;
}
