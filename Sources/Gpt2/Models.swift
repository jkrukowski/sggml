import Foundation
import Sggml

internal struct Params {
    internal var nVocab = 50257
    internal var nCtx = 1024
    internal var nEmbd = 768
    internal var nHead = 12
    internal var nLayer = 12
    internal var fType = 1
}

internal struct Layer {
    internal let ln_1_g: Tensor
    internal let ln_1_b: Tensor

    internal let ln_2_g: Tensor
    internal let ln_2_b: Tensor

    internal let c_attn_attn_w: Tensor
    internal let c_attn_attn_b: Tensor

    internal let c_attn_proj_w: Tensor
    internal let c_attn_proj_b: Tensor

    internal let c_mlp_fc_w: Tensor
    internal let c_mlp_fc_b: Tensor

    internal let c_mlp_proj_w: Tensor
    internal let c_mlp_proj_b: Tensor
}

internal struct Model {
    internal let params: Params
    internal let vocab: Vocab

    // normalization
    internal let ln_f_g: Tensor
    internal let ln_f_b: Tensor

    // position embedding
    internal let wte: Tensor
    // token embedding
    internal let wpe: Tensor
    // language model head
    internal let lm_head: Tensor

    internal let layers: [Layer]

    // key + value memory
    internal let memory_k: Tensor
    internal let memory_v: Tensor

    //
    internal let ctx: Context
    internal let tensors: [String: Tensor]
}

internal struct Vocab {
    internal var tokenToId: [String: Int]
    internal var idToToken: [Int: String]
    internal var specialTokens: [String]
}
