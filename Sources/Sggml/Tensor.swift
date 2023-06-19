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

    public var type: TensorType {
        guard let type = TensorType(rawValue: ggmlTensor.pointee.type) else {
            fatalError("Wrong tensor type \(ggmlTensor.pointee.type)")
        }
        return type
    }

    public var data: UnsafeMutableRawPointer {
        ggmlTensor.pointee.data
    }

    public var byteCount: Int {
        ggml_nbytes(ggmlTensor)
    }

    public var elementSize: Int {
        ggml_element_size(ggmlTensor)
    }

    internal let ggmlTensor: UnsafeMutablePointer<ggml_tensor>
    internal let ggmlContext: OpaquePointer

    internal init(
        ggmlContext: OpaquePointer,
        ggmlTensor: UnsafeMutablePointer<ggml_tensor>
    ) {
        self.ggmlTensor = ggmlTensor
        self.ggmlContext = ggmlContext
    }

    internal init(
        ggmlContext: OpaquePointer,
        type: TensorType,
        shape: [Int]
    ) throws {
        guard let ggmlTensor = ggml_new_tensor(ggmlContext, type.ggmlType, Int32(shape.count), shape.map { Int64($0) }) else {
            throw Error.failedToCreateTensor
        }
        self.init(ggmlContext: ggmlContext, ggmlTensor: ggmlTensor)
    }
}

extension Tensor {
    public func numberOfBytes(at index: Int) -> Int {
        assert(index < 4 && index >= 0)
        switch index {
        case 0:
            return Int(ggmlTensor.pointee.nb.0)
        case 1:
            return Int(ggmlTensor.pointee.nb.1)
        case 2:
            return Int(ggmlTensor.pointee.nb.2)
        case 3:
            return Int(ggmlTensor.pointee.nb.3)
        default:
            fatalError("Index out of range")
        }
    }

    public func numberOfElements(at index: Int) -> Int {
        assert(index < 4 && index >= 0)
        switch index {
        case 0:
            return Int(ggmlTensor.pointee.ne.0)
        case 1:
            return Int(ggmlTensor.pointee.ne.1)
        case 2:
            return Int(ggmlTensor.pointee.ne.2)
        case 3:
            return Int(ggmlTensor.pointee.ne.3)
        default:
            fatalError("Index out of range")
        }
    }

    public func numberOfElements() -> Int {
        Int(ggml_nelements(ggmlTensor))
    }

    public func copy<T>(from data: [T]) {
        ggmlTensor.pointee.data.copyMemory(from: data, byteCount: byteCount)
    }

    public func copy(from tensor: Tensor) {
        memcpy(data, tensor.data, tensor.byteCount)
    }

    public func data<Element>(count: Int) throws -> [Element] {
        guard let pointer = ggml_get_data(ggmlTensor) else {
            throw Error.failedToGetData
        }
        let bindedPointer = pointer.bindMemory(to: Element.self, capacity: count)
        return Array(UnsafeBufferPointer(start: bindedPointer, count: count))
    }
}
