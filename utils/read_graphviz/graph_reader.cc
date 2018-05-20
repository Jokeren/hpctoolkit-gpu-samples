#include "graph_reader.h"
#include <vector>

void GraphReader::read(Graph &graph) {
  // Read dot graph
  std::ifstream file(_file_name);
  std::stringstream dotfile;

  dotfile << file.rdbuf();
  file.close();

  boost::read_graphviz_detail::parser_result result;
  boost::read_graphviz_detail::parse_graphviz_from_string(dotfile.str(), result, true);

  std::unordered_map<std::string, size_t> vertex_name_to_id;
  read_vertices(result, vertex_name_to_id, graph);  
  read_edges(result, vertex_name_to_id, graph);
}


void GraphReader::read_vertices(
  const boost::read_graphviz_detail::parser_result &result,
  std::unordered_map<std::string, size_t> &vertex_name_to_id,
  Graph &graph) {
  size_t vertex_id = 0;
  for (auto node : result.nodes) {
    const std::string &vertex_name = node.first;
    const std::string &vertex_label = (node.second)["label"];
    Vertex *vertex = new Vertex(vertex_id, vertex_name, vertex_label);
    graph.vertices[vertex_id]->id = vertex_id;
    vertex_name_to_id[vertex_name] = vertex_id;
    ++vertex_id;
  }
}


void GraphReader::read_edges(
  const boost::read_graphviz_detail::parser_result &result,
  std::unordered_map<std::string, size_t> &vertex_name_to_id,
  Graph &graph) {
  for (auto einfo : result.edges) {
    size_t source_id = vertex_name_to_id[einfo.source.name];
    size_t target_id = vertex_name_to_id[einfo.target.name];
    std::vector<std::string> &source_port = einfo.source.location;
    std::vector<std::string> &target_port = einfo.target.location;
    graph.edges.push_back(new Edge(source_id, target_id, source_port, target_port));
  }
}
