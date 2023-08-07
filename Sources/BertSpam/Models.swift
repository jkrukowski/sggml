import Foundation
import Sggml

internal struct Params {
    internal var nVocab = 30522
    internal var nMaxTokens = 512
    internal var nEmbd = 256
    internal var nIntermediate = 1536
    internal var nHead = 12
    internal var nLayer = 6
    internal var fType = 1
}

internal struct Layer {
    internal let ln_att_w: Tensor
    internal let ln_att_b: Tensor

    internal let ln_out_w: Tensor
    internal let ln_out_b: Tensor

    internal let q_w: Tensor
    internal let q_b: Tensor
    internal let k_w: Tensor
    internal let k_b: Tensor
    internal let v_w: Tensor
    internal let v_b: Tensor

    internal let o_w: Tensor
    internal let o_b: Tensor

    internal let ff_i_w: Tensor
    internal let ff_i_b: Tensor

    internal let ff_o_w: Tensor
    internal let ff_o_b: Tensor
}

internal struct Model {
    internal let params: Params

    internal let wordEmbeddings: Tensor
    internal let tokenTypeEmbeddings: Tensor
    internal let positionEmbeddings: Tensor
    internal let ln_e_w: Tensor
    internal let ln_e_b: Tensor

    internal let layers: [Layer]

    internal let pc_d_w: Tensor
    internal let pc_d_b: Tensor

    internal let c_d_w: Tensor
    internal let c_d_b: Tensor

    internal let ctx: Context
    internal let tensors: [String: Tensor]
}

internal enum PredictionConfig {
    case memoryPerToken
    case prediction(inputEmbeddings: [Int], memoryPerToken: Int?)

    internal var inputEmbeddings: [Int] {
        switch self {
        case .memoryPerToken:
            return [0, 1, 2, 3]
        case let .prediction(inputEmbeddings, _):
            return inputEmbeddings
        }
    }

    internal var memoryPerToken: Int? {
        switch self {
        case .memoryPerToken:
            return nil
        case let .prediction(_, memoryPerToken):
            return memoryPerToken
        }
    }
}

internal enum PredictionResult {
    case memoryPerToken(Int)
    case prediction([Float])

    internal var memoryPerToken: Int? {
        switch self {
        case let .memoryPerToken(memoryPerToken):
            return memoryPerToken
        case .prediction:
            return nil
        }
    }

    internal var prediction: [Float]? {
        switch self {
        case .memoryPerToken:
            return nil
        case let .prediction(prediction):
            return prediction
        }
    }
}
