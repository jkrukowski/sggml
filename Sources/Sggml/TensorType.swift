import ggml

public enum TensorType {
    case f32
    case f16
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
