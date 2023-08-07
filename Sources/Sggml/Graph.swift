import ggml

public struct Graph {
    internal var ggmlGraph: ggml_cgraph
    internal var numberOfThreads: Int

    public init(numberOfThreads: Int) {
        ggmlGraph = ggml_cgraph()
        self.numberOfThreads = numberOfThreads
    }

    public mutating func buildForwardExpand(for tensor: Tensor) {
        ggml_build_forward_expand(&ggmlGraph, tensor.ggmlTensor)
    }

    public mutating func print() {
        ggml_graph_print(&ggmlGraph)
    }

    public mutating func saveToDotFile(at path: String) {
        ggml_graph_dump_dot(&ggmlGraph, nil, path)
    }
}
