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
        try Tensor(ggmlContext: ggmlContext, type: type, shape: shape)
    }
}

extension Context {
    public func matMul(_ t1: Tensor, _ t2: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_mul_mat(ggmlContext, t1.ggmlTensor, t2.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func add(_ t1: Tensor, _ t2: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_add(ggmlContext, t1.ggmlTensor, t2.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func relu(_ t1: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_relu(ggmlContext, t1.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func softmax(_ t1: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_soft_max(ggmlContext, t1.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func norm(_ t1: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_norm(ggmlContext, t1.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func rows(_ t1: Tensor, _ t2: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_get_rows(ggmlContext, t1.ggmlTensor, t2.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func `repeat`(_ t1: Tensor, _ t2: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_repeat(ggmlContext, t1.ggmlTensor, t2.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func permute(
        _ t1: Tensor,
        _ axis0: Int,
        _ axis1: Int,
        _ axis2: Int,
        _ axis3: Int
    ) throws -> Tensor {
        guard let ggmlTensor = ggml_permute(ggmlContext, t1.ggmlTensor, Int32(axis0), Int32(axis1), Int32(axis2), Int32(axis3)) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }

    public func copy(_ t1: Tensor, _ t2: Tensor) throws -> Tensor {
        guard let ggmlTensor = ggml_cpy(ggmlContext, t1.ggmlTensor, t2.ggmlTensor) else {
            throw Error.failedToCreateTensor
        }
        return Tensor(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }
}
