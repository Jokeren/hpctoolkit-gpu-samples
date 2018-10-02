#include <iostream>
#include <sstream>
#include <vector>
#include "cfg.h"
#include "graph.h"
#include "cfg_analyzer.h"
#include "cfg_parser.h"
#include "graph_reader.h"

int main(int argc, char *argv[]) {
  Graph graph;
  CFGParser cfg_parser;
  std::vector<Function *> functions;

  std::string file_name = std::string(argv[1]);
  GraphReader graph_reader(file_name);
  graph_reader.read(graph);
  cfg_parser.parse(graph, functions);

  std::cout << "Block info" << std::endl;
  for (auto function : functions) {
    std::cout << "Func: " << function->name << std::endl;
    for (auto block : function->blocks) {
      std::cout << "Block: " << block->name << std::endl;
      for (auto inst : block->insts) {
        std::cout << "Dual: " << inst->dual << " ";
        std::cout << "Port: " << inst->port << " ";
        std::cout << "Inst: ";
        std::cout << "<" << inst->offset << "> ";
        std::cout << inst->predicate << " ";
        std::cout << inst->opcode << " ";
        for (auto op : inst->operands) {
          std::cout << "[" << op << "]";
        }
        std::cout << std::endl;
      }
    }
  }

  std::cout << "Call info" << std::endl;
  CFGAnalyzer analyzer(functions);
  std::vector<Call *> calls = analyzer.extract_calls();
  for (auto call : calls) {
    std::stringstream stream;
    stream << std::hex << call->inst->offset;
    std::cout << "From func<" << call->caller_function->name << "> 0x" << stream.str() <<
      " to " << call->callee_function->name << std::endl;
  }

  std::cout << std::endl << "Loop info" << std::endl;
  std::vector<Loop *> loops = analyzer.extract_loops();
  for (auto loop : loops) {
    for (auto entry : loop->entries) {
      std::cout << "In function <" << loop->function->name << ">" << std::endl;
      std::cout << "Entry block <" << entry->entry_block->name << ">" <<  std::endl;
      std::cout << "Back edge block <" << entry->back_edge_block->name << ">" << std::endl;
      std::stringstream stream1, stream2;
      stream1 << std::hex << entry->entry_block->insts.front()->offset;
      std::cout << "Loop entry point 0x" << stream1.str() << std::endl;
      stream2 << std::hex << entry->back_edge_block->insts.back()->offset;
      std::cout << "Loop entry back edge 0x" << stream2.str() << std::endl;
      std::cout << "Loop blocks :" << std::endl;
      for (auto block : loop->blocks) {
        std::cout << block->name << std::endl;
      }
      std::cout << "Loop child blocks :" << std::endl;
      for (auto block : loop->child_blocks) {
        std::cout << block->name << std::endl;
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
