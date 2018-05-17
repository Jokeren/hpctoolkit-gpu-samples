#ifndef _GRAPH_READER_H_
#define _GRAPH_READER_H_

#include <vector>
#include <unordered_map>
#include <string>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/iteration_macros.hpp>
#include "cfg.h"

class GraphReader {
 public:
  GraphReader(const std::string &file_name) : _file_name(file_name) {}

  void construct_cfg();

  std::vector<Call *> analyze_calls();

  std::vector<Loop *> analyze_loops();

 private:
  struct Vertex {
    std::string name;
    std::string font_name;
    int font_size;
    std::string shape;
    std::string label;
  };

  struct Edge {
    std::string style;
  };

  // Create a typedef for the Graph type
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
          Vertex, Edge, boost::property<boost::graph_name_t, std::string> > Graph;

 private:
  void construct_blocks();
  void construct_functions();
  size_t find_block_parents(size_t node);
  void unite_blocks(size_t l, size_t r);

 private:
  Graph _g;
  std::string _file_name;
  std::vector<Block *> _blocks;
  std::vector<Function *> _functions;
  std::unordered_map<size_t, size_t> _block_parents;
};

#endif
