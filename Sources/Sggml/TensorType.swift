import ggml

public enum TensorType {
    case f32
    case f16

    case q4_0
    case q4_1

    case q5_0
    case q5_1

    case q8_0
    case q8_1

    case i8
    case i16
    case i32
    case count
}

extension TensorType {
    public init?(_ int: Int) {
        self.init(rawValue: ggml_type(rawValue: UInt32(int)))
    }
}

extension TensorType {
    public init?(rawValue: ggml_type) {
        switch rawValue {
        case GGML_TYPE_F32:
            self = .f32
        case GGML_TYPE_F16:
            self = .f16
        case GGML_TYPE_Q4_0:
            self = .q4_0
        case GGML_TYPE_Q4_1:
            self = .q4_1
        case GGML_TYPE_Q5_0:
            self = .q5_0
        case GGML_TYPE_Q5_1:
            self = .q5_1
        case GGML_TYPE_Q8_0:
            self = .q8_0
        case GGML_TYPE_Q8_1:
            self = .q8_1
        case GGML_TYPE_I8:
            self = .i8
        case GGML_TYPE_I16:
            self = .i16
        case GGML_TYPE_I32:
            self = .i32
        case GGML_TYPE_COUNT:
            self = .count
        default:
            return nil
        }
    }
}

extension TensorType {
    public var typeSize: Float {
        ggml_type_sizef(ggmlType)
    }

    public var typeSizeInt: Int {
        ggml_type_size(ggmlType)
    }

    public var blockSize: Int {
        Int(ggml_blck_size(ggmlType))
    }
}

extension TensorType {
    public init?(fType: Int) {
        let ggmlType = ggml_ftype_to_ggml_type(ggml_ftype(Int32(fType)))
        self.init(rawValue: ggmlType)
    }
}

extension TensorType {
    internal var ggmlType: ggml_type {
        switch self {
        case .f32:
            return GGML_TYPE_F32
        case .f16:
            return GGML_TYPE_F16
        case .q4_0:
            return GGML_TYPE_Q4_0
        case .q4_1:
            return GGML_TYPE_Q4_1
        case .q5_0:
            return GGML_TYPE_Q5_0
        case .q5_1:
            return GGML_TYPE_Q5_1
        case .q8_0:
            return GGML_TYPE_Q8_0
        case .q8_1:
            return GGML_TYPE_Q8_1
        case .i8:
            return GGML_TYPE_I8
        case .i16:
            return GGML_TYPE_I16
        case .i32:
            return GGML_TYPE_I32
        case .count:
            return GGML_TYPE_COUNT
        }
    }
}
