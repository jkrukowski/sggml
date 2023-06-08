import ggml

public enum TensorType {
    case f32
    case f16
}

extension TensorType {
    public init?(rawValue: ggml_type) {
        switch rawValue {
        case GGML_TYPE_F32:
            self = .f32
        case GGML_TYPE_F16:
            self = .f16
        default:
            return nil
        }
    }
}

extension TensorType {
    public var typeSize: Float {
        ggml_type_sizef(ggmlType)
    }
}

extension TensorType {
    internal var ggmlType: ggml_type {
        switch self {
        case .f32:
            return GGML_TYPE_F32
        case .f16:
            return GGML_TYPE_F16
        }
    }
}
