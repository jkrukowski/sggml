import Foundation
import ggml
import Logging
import Sggml
import Utils

private let logger = Logger(label: "bert-spam")

private func readParams(stream: InputStream) -> Params {
    let nVocab: Int32 = stream.read()
    let nMaxTokens: Int32 = stream.read()
    let nEmbd: Int32 = stream.read()
    let nIntermediate: Int32 = stream.read()
    let nHead: Int32 = stream.read()
    let nLayer: Int32 = stream.read()
    let fType: Int32 = stream.read()
    return Params(
        nVocab: Int(nVocab),
        nMaxTokens: Int(nMaxTokens),
        nEmbd: Int(nEmbd),
        nIntermediate: Int(nIntermediate),
        nHead: Int(nHead),
        nLayer: Int(nLayer),
        fType: Int(fType)
    )
}

private func contextSize(
    params: Params,
    wType: TensorType
) -> Float {
    var contextSize: Float = 0.0
    let nEmbd = Float(params.nEmbd)
    let nVocab = Float(params.nVocab)
    let nMaxTokens = Float(params.nMaxTokens)
    let nLayer = Float(params.nLayer)
    let nIntermediate = Float(params.nIntermediate)

    contextSize += nEmbd * nVocab * wType.typeSize // word_embeddings
    contextSize += nEmbd * 2 * wType.typeSize // token_type_embeddings
    contextSize += nEmbd * nMaxTokens * wType.typeSize // position_embeddings

    contextSize += 2 * nEmbd * TensorType.f32.typeSize // ln_e_*

    contextSize += 4 * nLayer * (nEmbd * TensorType.f32.typeSize) // ln_*

    contextSize += 4 * nLayer * (nEmbd * nEmbd * wType.typeSize) // kqvo weights
    contextSize += 4 * nLayer * (nEmbd * TensorType.f32.typeSize) // kqvo bias

    contextSize += 2 * nLayer * (nEmbd * nIntermediate * wType.typeSize) // ff_*_w
    contextSize += nLayer * (nIntermediate * TensorType.f32.typeSize) // ff_i_b
    contextSize += nLayer * (nEmbd * TensorType.f32.typeSize) // ff_o_b

    contextSize += nEmbd * nEmbd * wType.typeSize // p_c weights
    contextSize += nEmbd * TensorType.f32.typeSize // p_c bias

    contextSize += 2 * nEmbd * wType.typeSize // c_d weights
    contextSize += 2 * TensorType.f32.typeSize // c_d bias

    contextSize += (6 + 5 + 16 * nLayer) * 512 // object overhead
    return contextSize
}

private func loadModelData(
    stream: InputStream,
    params: Params,
    contextSize: Int,
    wType: TensorType
) throws -> Model {
    let model = try createModel(
        params: params,
        contextSize: Int(contextSize),
        wType: wType
    )

    var totalSize = 0
    while stream.hasBytesAvailable {
        let nDims = Int(stream.read() as Int32)
        let length = Int(stream.read() as Int32)
        let fType = Int(stream.read() as Int32)

        if !stream.hasBytesAvailable {
            break
        }

        var nElements = 1
        var ne = [Int32](repeating: 1, count: 2)
        for i in 0 ..< nDims {
            ne[i] = stream.read()
            nElements *= Int(ne[i])
        }

        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: length)
        defer { buffer.deallocate() }
        stream.read(buffer, maxLength: length)
        guard let name = String(bytes: UnsafeBufferPointer(start: buffer, count: length), encoding: .ascii) else {
            logger.error("Invalid name")
            throw Error.invalidFile
        }

        guard let tensor = model.tensors[name] else {
            logger.error("Unknown tensor \(name)")
            throw Error.invalidFile
        }

        guard tensor.numberOfElements() == nElements else {
            logger.error("Invalid number of elements \(tensor.numberOfElements()) != \(nElements) \(name)")
            throw Error.invalidFile
        }

        guard tensor.numberOfElements(at: 0) == ne[0] else {
            logger.error("Invalid number of elements at 0 \(tensor.numberOfElements(at: 0)) != \(ne[0])")
            throw Error.invalidFile
        }

        guard tensor.numberOfElements(at: 1) == ne[1] else {
            logger.error("Invalid number of elements at 1 \(tensor.numberOfElements(at: 1)) != \(ne[1])")
            throw Error.invalidFile
        }

        guard let bpe = TensorType(fType) else {
            logger.error("Invalid tensor type \(fType)")
            throw Error.invalidFile
        }

        let tensorType = tensor.type

        guard (nElements * bpe.typeSizeInt) / tensorType.blockSize == tensor.byteCount else {
            logger.error("Invalid tensor size")
            throw Error.invalidFile
        }

        let readLength = stream.read(tensor.data, maxLength: tensor.byteCount)
        assert(readLength == tensor.byteCount)
        totalSize += tensor.byteCount
    }
    logger.info("Model size: \(Float(totalSize) / (1024.0 * 1024.0)) MB")
    return model
}

private func createModel(
    params: Params,
    contextSize: Int,
    wType: TensorType
) throws -> Model {
    let context = try Context(memorySize: contextSize, noAlloc: false)

    var layers = [Layer]()
    layers.reserveCapacity(params.nLayer)
    var tensors = [String: Tensor]()

    let wordEmbeddings = try context.tensor(type: wType, shape: [params.nEmbd, params.nVocab])
    let tokenTypeEmbeddings = try context.tensor(type: wType, shape: [params.nEmbd, 2])
    let positionEmbeddings = try context.tensor(type: wType, shape: [params.nEmbd, params.nMaxTokens])

    let ln_e_w = try context.tensor(type: .f32, shape: [params.nEmbd])
    let ln_e_b = try context.tensor(type: .f32, shape: [params.nEmbd])

    let pc_d_w = try context.tensor(type: wType, shape: [params.nEmbd, params.nEmbd])
    let pc_d_b = try context.tensor(type: .f32, shape: [params.nEmbd])

    let c_d_w = try context.tensor(type: wType, shape: [params.nEmbd, 2])
    let c_d_b = try context.tensor(type: .f32, shape: [2])

    tensors["l1.embeddings.word_embeddings.weight"] = wordEmbeddings
    tensors["l1.embeddings.token_type_embeddings.weight"] = tokenTypeEmbeddings
    tensors["l1.embeddings.position_embeddings.weight"] = positionEmbeddings

    tensors["l1.embeddings.LayerNorm.weight"] = ln_e_w
    tensors["l1.embeddings.LayerNorm.bias"] = ln_e_b

    tensors["pre_classifier.weight"] = pc_d_w
    tensors["pre_classifier.bias"] = pc_d_b

    tensors["classifier.weight"] = c_d_w
    tensors["classifier.bias"] = c_d_b

    for index in 0 ..< params.nLayer {
        let layer = try Layer(
            ln_att_w: context.tensor(type: .f32, shape: [params.nEmbd]),
            ln_att_b: context.tensor(type: .f32, shape: [params.nEmbd]),
            ln_out_w: context.tensor(type: .f32, shape: [params.nEmbd]),
            ln_out_b: context.tensor(type: .f32, shape: [params.nEmbd]),
            q_w: context.tensor(type: wType, shape: [params.nEmbd, params.nEmbd]),
            q_b: context.tensor(type: .f32, shape: [params.nEmbd]),
            k_w: context.tensor(type: wType, shape: [params.nEmbd, params.nEmbd]),
            k_b: context.tensor(type: .f32, shape: [params.nEmbd]),
            v_w: context.tensor(type: wType, shape: [params.nEmbd, params.nEmbd]),
            v_b: context.tensor(type: .f32, shape: [params.nEmbd]),
            o_w: context.tensor(type: wType, shape: [params.nEmbd, params.nEmbd]),
            o_b: context.tensor(type: .f32, shape: [params.nEmbd]),
            ff_i_w: context.tensor(type: wType, shape: [params.nEmbd, params.nIntermediate]),
            ff_i_b: context.tensor(type: .f32, shape: [params.nIntermediate]),
            ff_o_w: context.tensor(type: wType, shape: [params.nIntermediate, params.nEmbd]),
            ff_o_b: context.tensor(type: .f32, shape: [params.nEmbd])
        )
        layers.append(layer)

        // map by name
        tensors["l1.encoder.layer.\(index).attention.self.query.weight"] = layer.q_w
        tensors["l1.encoder.layer.\(index).attention.self.query.bias"] = layer.q_b
        tensors["l1.encoder.layer.\(index).attention.self.key.weight"] = layer.k_w
        tensors["l1.encoder.layer.\(index).attention.self.key.bias"] = layer.k_b
        tensors["l1.encoder.layer.\(index).attention.self.value.weight"] = layer.v_w
        tensors["l1.encoder.layer.\(index).attention.self.value.bias"] = layer.v_b
        tensors["l1.encoder.layer.\(index).attention.output.LayerNorm.weight"] = layer.ln_att_w
        tensors["l1.encoder.layer.\(index).attention.output.LayerNorm.bias"] = layer.ln_att_b
        tensors["l1.encoder.layer.\(index).attention.output.dense.weight"] = layer.o_w
        tensors["l1.encoder.layer.\(index).attention.output.dense.bias"] = layer.o_b

        tensors["l1.encoder.layer.\(index).intermediate.dense.weight"] = layer.ff_i_w
        tensors["l1.encoder.layer.\(index).intermediate.dense.bias"] = layer.ff_i_b

        tensors["l1.encoder.layer.\(index).output.LayerNorm.weight"] = layer.ln_out_w
        tensors["l1.encoder.layer.\(index).output.LayerNorm.bias"] = layer.ln_out_b
        tensors["l1.encoder.layer.\(index).output.dense.weight"] = layer.ff_o_w
        tensors["l1.encoder.layer.\(index).output.dense.bias"] = layer.ff_o_b
    }

    return Model(
        params: params,
        wordEmbeddings: wordEmbeddings,
        tokenTypeEmbeddings: tokenTypeEmbeddings,
        positionEmbeddings: positionEmbeddings,
        ln_e_w: ln_e_w,
        ln_e_b: ln_e_b,
        layers: layers,
        pc_d_w: pc_d_w,
        pc_d_b: pc_d_b,
        c_d_w: c_d_w,
        c_d_b: c_d_b,
        ctx: context,
        tensors: tensors
    )
}

internal func readModel(at modelPath: URL) throws -> Model {
    guard let stream = InputStream(fileAtPath: modelPath.path()) else {
        throw Error.invalidFile
    }
    stream.open()
    defer { stream.close() }

    if !verifyMagic(stream: stream) {
        logger.error("Invalid magic number")
        throw Error.invalidFile
    }
    let params = readParams(stream: stream)
    let wType: TensorType
    switch params.fType {
    case 0:
        wType = .f32
    case 1:
        wType = .f16
    case 2:
        wType = .q4_0
    case 3:
        wType = .q4_1
    default:
        logger.error("Invalid fType \(params.fType)")
        throw Error.invalidFile
    }

    let contextSize = contextSize(params: params, wType: wType)
    logger.info("Context size: \(contextSize / (1024.0 * 1024.0)) MB")
    return try loadModelData(
        stream: stream,
        params: params,
        contextSize: Int(contextSize),
        wType: wType
    )
}

@discardableResult
internal func predict(
    model: Model,
    numberOfThreads: Int,
    config: PredictionConfig
) throws -> PredictionResult {
    let inputEmbeddings = config.inputEmbeddings
    let N = inputEmbeddings.count
    let nEmbd = model.params.nEmbd
    let nHead = model.params.nHead
    let dHead = nEmbd / nHead

    var bufferSize = 16 * 1024 * 1024
    switch config {
    case let .prediction(_, .some(memoryPerToken)) where 2 * memoryPerToken * N > bufferSize:
        bufferSize = Int(1.1 * Float(2 * memoryPerToken * N))
        logger.info("Reallocating buffer to \(bufferSize) bytes")
    default:
        break
    }
    let context = try Context(memorySize: bufferSize, noAlloc: false)
    var graph = Graph(numberOfThreads: numberOfThreads)

    let tokenLayer = try context.tensor(type: .i32, shape: [N])
    tokenLayer.copy(from: inputEmbeddings.map { Int32($0) })

    let tokenTypes = try context.zero(context.tensor(type: .i32, shape: [N]))

    let positions = try context.tensor(type: .i32, shape: [N])
    positions.copy(from: (0 ..< N).map { Int32($0) })

    var inpL = try context.rows(model.wordEmbeddings, tokenLayer)

    inpL = try context.add(
        context.rows(model.tokenTypeEmbeddings, tokenTypes),
        inpL
    )
    inpL = try context.add(
        context.rows(model.positionEmbeddings, positions),
        inpL
    )

    // embd norm
    inpL = try context.norm(inpL)
    inpL = try context.add(
        context.mul(
            context.repeat(model.ln_e_w, inpL),
            inpL
        ),
        context.repeat(model.ln_e_b, inpL)
    )

    // layers
    for index in 0 ..< model.params.nLayer {
        var cur = inpL

        // self-attention
        var Qcur = cur
        Qcur = try context.reshape3d(
            context.add(
                context.repeat(model.layers[index].q_b, Qcur),
                context.matMul(model.layers[index].q_w, Qcur)
            ),
            dHead,
            nHead,
            N
        )
        let Q = try context.permute(Qcur, 0, 2, 1, 3)

        var Kcur = cur
        Kcur = try context.reshape3d(
            context.add(
                context.repeat(model.layers[index].k_b, Kcur),
                context.matMul(model.layers[index].k_w, Kcur)
            ),
            dHead,
            nHead,
            N
        )
        let K = try context.permute(Kcur, 0, 2, 1, 3)

        var Vcur = cur
        Vcur = try context.reshape3d(
            context.add(
                context.repeat(model.layers[index].v_b, Vcur),
                context.matMul(model.layers[index].v_w, Vcur)
            ),
            dHead,
            nHead,
            N
        )
        var V = try context.permute(Vcur, 0, 2, 1, 3)

        var KQ = try context.matMul(K, Q)
        KQ = try context.softmax(
            context.scale(
                KQ,
                context.f32(1.0 / sqrt(Float(dHead)))
            )
        )

        V = try context.cont(
            context.transpose(V)
        )

        var KQV = try context.matMul(V, KQ)
        KQV = try context.permute(KQV, 0, 2, 1, 3)

        cur = try context.copy(
            KQV,
            context.tensor(type: .f32, shape: [nEmbd, N])
        )

        // attention output
        cur = try context.add(
            context.repeat(model.layers[index].o_b, cur),
            context.matMul(model.layers[index].o_w, cur)
        )

        // re-add the layer input
        cur = try context.add(cur, inpL)

        // attention norm
        cur = try context.norm(cur)

        cur = try context.add(
            context.mul(
                context.repeat(model.layers[index].ln_att_w, cur),
                cur
            ),
            context.repeat(model.layers[index].ln_att_b, cur)
        )

        let att_output = cur
        // intermediate_output = self.intermediate(attention_output)
        cur = try context.matMul(model.layers[index].ff_i_w, cur)
        cur = try context.add(
            context.repeat(model.layers[index].ff_i_b, cur),
            cur
        )
        cur = try context.gelu(cur)

        // layer_output = self.output(intermediate_output, attention_output)
        cur = try context.matMul(model.layers[index].ff_o_w, cur)
        cur = try context.add(
            context.repeat(model.layers[index].ff_o_b, cur),
            cur
        )

        // attentions bypass the intermediate layer
        cur = try context.add(att_output, cur)

        // output norm
        cur = try context.norm(cur)
        cur = try context.add(
            context.mul(
                context.repeat(model.layers[index].ln_out_w, cur),
                cur
            ),
            context.repeat(model.layers[index].ln_out_b, cur)
        )

        inpL = cur
    }

    // pre classifier
    inpL = try context.matMul(model.pc_d_w, inpL)
    inpL = try context.add(
        context.repeat(model.pc_d_b, inpL),
        inpL
    )
    inpL = try context.relu(inpL)

    // classifier
    inpL = try context.matMul(model.c_d_w, inpL)
    inpL = try context.add(
        context.repeat(model.c_d_b, inpL),
        inpL
    )

    inpL = try context.softmax(inpL)

    graph.buildForwardExpand(for: inpL)
    context.compute(graph: &graph)

    switch config {
    case .memoryPerToken:
        return .memoryPerToken(context.usedMemory / N)
    case .prediction:
        let ptr = inpL.data.assumingMemoryBound(to: Float.self)
        return .prediction(Array(UnsafeBufferPointer(start: ptr, count: 2)))
    }
}
