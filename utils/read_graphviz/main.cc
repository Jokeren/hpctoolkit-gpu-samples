#include <iostream>                  // for std::cout
#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include <deque>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <regex>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/iteration_macros.hpp>

using namespace boost;


struct Vertex {
  std::string name;
  std::string fontname;
  int fontsize;
  std::string shape;
  std::string label;
};


struct Edge {
  std::string style;
};


struct Inst {
  std::string offset;
  std::string operate;
  std::vector<std::string> operands;

  Inst(const std::string &inst_str) {
    std::istringstream ss(inst_str);
    std::string s;
    if (std::getline(ss, s, ':')) {
      this->offset = s;
      //std::cout << " offset: " << s << std::endl;
      if (std::getline(ss, s, ':')) {
        std::regex e("\\\\ ");
        ss = std::istringstream(std::regex_replace(s, e, "\n"));
        while (std::getline(ss, s)) {
          if (s != "") {
            if (this->operate == "") {
              this->operate = s;
              //std::cout << " operate: " << s << std::endl;
            } else {
              this->operands.push_back(s);
              //std::cout << " operands: " << s << std::endl;
            }
          }
        }
      }
    }
  }
};


std::map<size_t, std::vector<Inst> > insts;

int main() {
  // Create a typedef for the Graph type
  typedef adjacency_list<vecS, vecS, directedS, Vertex, Edge, property<graph_name_t, std::string> > Graph;

  // declare a graph object
  Graph g; //CAUTION: Graph g(); is a function declaration
  dynamic_properties dp;

  dp.property("node_id", get(&Vertex::name, g));
  dp.property("fontname", get(&Vertex::fontname, g));
  dp.property("fontsize", get(&Vertex::fontsize, g));
  dp.property("shape", get(&Vertex::shape, g));
  dp.property("label", get(&Vertex::label, g));
  dp.property("style",get(&Edge::style,g));

  boost::ref_property_map<Graph*,std::string> gname(get_property(g, graph_name));
  dp.property("graph_id", gname);

  std::ifstream file("sample.dot");
  std::stringstream dotfile;
  dotfile << file.rdbuf();
  file.close();

  read_graphviz(dotfile, g, dp);

  std::regex e("\\\\l([|]*)");
  std::cout << std::endl << "Every vertex pos:" << std::endl;
  BGL_FORALL_VERTICES(v, g, Graph)
  {
    std::cout << v << ": " << g[v].name << std::endl;
    std::string label = g[v].label;
    std::istringstream ss(std::regex_replace(label, e, "\n"));
    
    std::deque<std::string> inst_strings;
    std::string s;
    while (std::getline(ss, s)) {
      inst_strings.push_back(s);
    }
    inst_strings.pop_front();
    inst_strings.pop_back();

    for (auto inst_string : inst_strings) {
      insts[v].push_back(Inst(inst_string));
    }
  }

  std::cout << std::endl << "Every edge pos:" << std::endl;
  BGL_FORALL_EDGES(e, g, Graph)
  {
    std::cout << g[source(e,g)].name << " -> " << g[target(e,g)].name << ": " << g[e].style << std::endl;
  }

  std::cout << std::endl << "Every call:" << std::endl;
  for (auto entry : insts) {
    for (auto inst : entry.second) {
      if (inst.operate.find("CALL") != std::string::npos) {
        std::string &operand = inst.operands[0];
        std::string callee = operand.substr(2, operand.size() - 4);
        std::cout << "From 0x" << inst.offset << " to " << callee << std::endl;
      }
    }
  }

  return 0;
}
