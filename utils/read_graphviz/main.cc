#include <iostream>                  // for std::cout
#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include <string>
#include <sstream>
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

  std::cout << "Every vertex pos:" << std::endl;
  BGL_FORALL_VERTICES(v,g,Graph)
  {
    std::cout << v << ": " << g[v].name << std::endl;
  }

  std::cout << "Every edge pos:" << std::endl;
  BGL_FORALL_EDGES(e,g,Graph)
  {
    std::cout << g[source(e,g)].name << " -> " << g[target(e,g)].name << ": " << g[e].style << std::endl;
  }

  return 0;
}
