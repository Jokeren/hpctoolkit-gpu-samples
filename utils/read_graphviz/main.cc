#include <iostream>
#include <sstream>
#include <vector>
#include "cfg.h"
#include "graph_reader.h"

int main() {
  GraphReader graph_reader("sample.dot");
  graph_reader.construct_cfg();

  std::vector<Call *> calls = graph_reader.analyze_calls();
  for (auto call : calls) {
    std::stringstream stream;
    stream << std::hex << call->caller_block->insts[0]->offset;
    std::cout << "From func<" << call->caller_function->name << "> 0x" << stream.str() <<
      " to " << call->callee_function->name << std::endl;
  }

  std::vector<Loop *> loops = graph_reader.analyze_loops();

  return 0;
}
