import ggml

internal final class ContextWrapper {
    internal let ggmlContext: OpaquePointer

    internal init(_ ggmlContext: OpaquePointer) {
        self.ggmlContext = ggmlContext
    }

    deinit {
        ggml_free(ggmlContext)
    }
}

public struct Context {
    internal let contextWrapper: ContextWrapper
    private var ggmlContext: OpaquePointer {
        contextWrapper.ggmlContext
    }

    public init(memorySize: Int, noAlloc: Bool) throws {
        let contextParams = ggml_init_params(
            mem_size: memorySize,
            mem_buffer: nil, // TODO: add a param
            no_alloc: noAlloc
        )
        guard let ggmlContext = ggml_init(contextParams) else {
            throw Error.failedToCreateContext
        }
        contextWrapper = ContextWrapper(ggmlContext)
    }
}

extension Context {
    public func compute(graph: inout Graph) {
        ggml_graph_compute(ggmlContext, &graph.ggmlGraph)
    }
}

extension Context {
    public func tensor(type: TensorType, shape: [Int]) throws -> Tensor {
        try Tensor(ggmlContext: ggmlContext, type: type, shape: shape.map { Int64($0) })
    }
}

extension Context {
    public func matMul(_ t1: Tensor, _ t2: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_mul_mat(ggmlContext, t1.ggmlTensor, t2.ggmlTensor) else {
            throw Error.failedToCreateTesnor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func add(_ t1: Tensor, _ t2: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_add(ggmlContext, t1.ggmlTensor, t2.ggmlTensor) else {
            throw Error.failedToCreateTesnor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func relu(_ t1: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_relu(ggmlContext, t1.ggmlTensor) else {
            throw Error.failedToCreateTesnor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func softmax(_ t1: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_soft_max(ggmlContext, t1.ggmlTensor) else {
            throw Error.failedToCreateTesnor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }
}
