import ggml

public struct Tensor {
    public var name: String {
        set {
            ggml_set_name(ggmlTensor, newValue)
        }
        get {
            String(cString: ggml_get_name(ggmlTensor))
        }
    }

    public var data: UnsafeMutableRawPointer {
        ggmlTensor.pointee.data
    }

    public var maxLength: Int {
        Int(ggml_nbytes(ggmlTensor))
    }

    internal let ggmlTensor: UnsafeMutablePointer<ggml_tensor>
    internal let ggmlContext: OpaquePointer

    internal init(ggmlContext: OpaquePointer, ggmlTensor: UnsafeMutablePointer<ggml_tensor>) {
        self.ggmlTensor = ggmlTensor
        self.ggmlContext = ggmlContext
    }

    internal init(ggmlContext: OpaquePointer, type: TensorType, shape: [Int64]) throws {
        guard let ggmlTensor = ggml_new_tensor(ggmlContext, type.ggmlType, Int32(shape.count), shape) else {
            throw Error.failedToCreateTesnor
        }
        self.init(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }
}

extension Tensor {
    public func copy(from data: [Float]) {
        ggmlTensor.pointee.data.copyMemory(from: data, byteCount: ggml_nbytes(ggmlTensor))
    }

    public func data<Element>(count: Int) throws -> [Element] {
        guard let pointer = ggml_get_data(ggmlTensor) else {
            throw Error.failedToGetData
        }
        let bindedPointer = pointer.bindMemory(to: Element.self, capacity: count)
        return Array(UnsafeBufferPointer(start: bindedPointer, count: count))
    }
}
