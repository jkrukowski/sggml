import BaseMath
import CBaseMath
import Foundation
import ggml
import Logging
import Sggml
import Utils

private let logger = Logger(label: "sggml")

private func readParams(stream: InputStream) -> Params {
    let nVocab: Int32 = stream.read()
    let nCtx: Int32 = stream.read()
    let nEmbd: Int32 = stream.read()
    let nHead: Int32 = stream.read()
    let nLayer: Int32 = stream.read()
    let fType: Int32 = stream.read()
    return Params(
        nVocab: Int(nVocab),
        nCtx: Int(nCtx),
        nEmbd: Int(nEmbd),
        nHead: Int(nHead),
        nLayer: Int(nLayer),
        fType: Int(fType) % Constants.qntVersionFactor
    )
}

private func readVocab(
    stream: InputStream,
    params: Params
) throws -> Vocab {
    let nVocab = Int(stream.read() as Int32)
    if nVocab != params.nVocab {
        logger.error("Invalid vocabulary size")
        throw Error.invalidFile
    }
    var tokenToId = [String: Int]()
    var idToToken = [Int: String]()
    for index in 0 ..< nVocab {
        let length = Int(stream.read() as Int32)
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: length)
        defer { buffer.deallocate() }
        stream.read(buffer, maxLength: length)
        let token = String(cString: buffer)
        tokenToId[token] = index
        idToToken[index] = token
    }
    return Vocab(
        tokenToId: tokenToId,
        idToToken: idToToken,
        specialTokens: []
    )
}

private func contextSize(
    params: Params,
    wType: TensorType
) -> Float {
    var contextSize: Float = 0.0
    let nEmbd = Float(params.nEmbd)
    let nLayer = Float(params.nLayer)
    let nCtx = Float(params.nCtx)
    let nVocab = Float(params.nVocab)

    contextSize += nEmbd * TensorType.f32.typeSize // ln_f_g
    contextSize += nEmbd * TensorType.f32.typeSize // ln_f_b

    contextSize += nVocab * nEmbd * wType.typeSize // wte
    contextSize += nCtx * nEmbd * TensorType.f32.typeSize // wpe
    contextSize += nVocab * nEmbd * wType.typeSize // lm_head

    contextSize += nLayer * nEmbd * TensorType.f32.typeSize // ln_1_g
    contextSize += nLayer * nEmbd * TensorType.f32.typeSize // ln_1_b

    contextSize += nLayer * nEmbd * TensorType.f32.typeSize // ln_2_g
    contextSize += nLayer * nEmbd * TensorType.f32.typeSize // ln_2_b

    contextSize += nLayer * (3 * nEmbd * nEmbd * wType.typeSize) // c_attn_attn_w
    contextSize += nLayer * (3 * nEmbd * TensorType.f32.typeSize) // c_attn_attn_b

    contextSize += nLayer * (nEmbd * nEmbd * wType.typeSize) // c_attn_proj_w
    contextSize += nLayer * (nEmbd * TensorType.f32.typeSize) // c_attn_proj_b

    contextSize += nLayer * (4 * nEmbd * nEmbd * wType.typeSize) // c_mlp_fc_w
    contextSize += nLayer * (4 * nEmbd * TensorType.f32.typeSize) // c_mlp_fc_b

    contextSize += nLayer * (4 * nEmbd * nEmbd * wType.typeSize) // c_mlp_proj_w
    contextSize += nLayer * (nEmbd * TensorType.f32.typeSize) // c_mlp_proj_b

    contextSize += nCtx * nLayer * nEmbd * TensorType.f32.typeSize // memory_k
    contextSize += nCtx * nLayer * nEmbd * TensorType.f32.typeSize // memory_v

    contextSize += (6.0 + 12.0 * nLayer) * 512.0 // object overhead
    return contextSize
}

private func createModel(
    params: Params,
    vocab: Vocab,
    contextSize: Int,
    wType: TensorType
) throws -> Model {
    let context = try Context(memorySize: contextSize, noAlloc: false)

    var layers = [Layer]()
    layers.reserveCapacity(params.nLayer)
    var tensors = [String: Tensor]()

    let ln_f_g = try context.tensor(type: TensorType.f32, shape: [params.nEmbd])
    let ln_f_b = try context.tensor(type: TensorType.f32, shape: [params.nEmbd])

    let wte = try context.tensor(type: wType, shape: [params.nEmbd, params.nVocab])
    let wpe = try context.tensor(type: TensorType.f32, shape: [params.nEmbd, params.nCtx])
    let lm_head = try context.tensor(type: wType, shape: [params.nEmbd, params.nVocab])

    tensors["model/ln_f/g"] = ln_f_g
    tensors["model/ln_f/b"] = ln_f_b

    tensors["model/wte"] = wte
    tensors["model/wpe"] = wpe
    tensors["model/lm_head"] = lm_head

    for i in 0 ..< params.nLayer {
        let layer = try Layer(
            ln_1_g: context.tensor(type: TensorType.f32, shape: [params.nEmbd]),
            ln_1_b: context.tensor(type: TensorType.f32, shape: [params.nEmbd]),
            ln_2_g: context.tensor(type: TensorType.f32, shape: [params.nEmbd]),
            ln_2_b: context.tensor(type: TensorType.f32, shape: [params.nEmbd]),
            c_attn_attn_w: context.tensor(type: wType, shape: [params.nEmbd, 3 * params.nEmbd]),
            c_attn_attn_b: context.tensor(type: TensorType.f32, shape: [3 * params.nEmbd]),
            c_attn_proj_w: context.tensor(type: wType, shape: [params.nEmbd, params.nEmbd]),
            c_attn_proj_b: context.tensor(type: TensorType.f32, shape: [params.nEmbd]),
            c_mlp_fc_w: context.tensor(type: wType, shape: [params.nEmbd, 4 * params.nEmbd]),
            c_mlp_fc_b: context.tensor(type: TensorType.f32, shape: [4 * params.nEmbd]),
            c_mlp_proj_w: context.tensor(type: wType, shape: [4 * params.nEmbd, params.nEmbd]),
            c_mlp_proj_b: context.tensor(type: TensorType.f32, shape: [params.nEmbd])
        )
        layers.append(layer)
        tensors["model/h\(i)/ln_1/g"] = layer.ln_1_g
        tensors["model/h\(i)/ln_1/b"] = layer.ln_1_b

        tensors["model/h\(i)/ln_2/g"] = layer.ln_2_g
        tensors["model/h\(i)/ln_2/b"] = layer.ln_2_b

        tensors["model/h\(i)/attn/c_attn/w"] = layer.c_attn_attn_w
        tensors["model/h\(i)/attn/c_attn/b"] = layer.c_attn_attn_b

        tensors["model/h\(i)/attn/c_proj/w"] = layer.c_attn_proj_w
        tensors["model/h\(i)/attn/c_proj/b"] = layer.c_attn_proj_b

        tensors["model/h\(i)/mlp/c_fc/w"] = layer.c_mlp_fc_w
        tensors["model/h\(i)/mlp/c_fc/b"] = layer.c_mlp_fc_b

        tensors["model/h\(i)/mlp/c_proj/w"] = layer.c_mlp_proj_w
        tensors["model/h\(i)/mlp/c_proj/b"] = layer.c_mlp_proj_b
    }

    let nMem = params.nLayer * params.nCtx
    let nElement = params.nEmbd * nMem

    let memoryK = try context.tensor(type: TensorType.f32, shape: [nElement])
    let memoryV = try context.tensor(type: TensorType.f32, shape: [nElement])

    return Model(
        params: params,
        vocab: vocab,
        ln_f_g: ln_f_g,
        ln_f_b: ln_f_b,
        wte: wte,
        wpe: wpe,
        lm_head: lm_head,
        layers: layers,
        memory_k: memoryK,
        memory_v: memoryV,
        ctx: context,
        tensors: tensors
    )
}

private func loadModelData(
    stream: InputStream,
    params: Params,
    vocab: Vocab,
    contextSize: Int,
    wType: TensorType
) throws -> Model {
    let model = try createModel(params: params, vocab: vocab, contextSize: contextSize, wType: wType)

    var totalSize = 0
    var hasLmHead = false
    while stream.hasBytesAvailable {
        let nDims = Int(stream.read() as Int32)
        let length = Int(stream.read() as Int32)
        let tType = Int(stream.read() as Int32)

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

        guard let bpe = TensorType(tType) else {
            logger.error("Invalid tensor type \(tType)")
            throw Error.invalidFile
        }

        let tensorType = tensor.type

        guard (nElements * bpe.typeSizeInt) / tensorType.blockSize == tensor.byteCount else {
            logger.error("Invalid tensor size")
            throw Error.invalidFile
        }

        let readLength = stream.read(tensor.data, maxLength: tensor.byteCount)
        assert(readLength == tensor.byteCount)

        if name == "model/wte", !hasLmHead {
            model.lm_head.copy(from: tensor)
        }
        if name == "model/lm_head" {
            hasLmHead = true
        }
        totalSize += tensor.byteCount
    }
    logger.info("Model size: \(Float(totalSize) / (1024.0 * 1024.0)) MB")
    return model
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
    let vocab = try readVocab(stream: stream, params: params)
    guard let wType = TensorType(fType: params.fType) else {
        throw Error.invalidFile
    }
    if wType == .count {
        logger.error("Bad fType value \(params.fType)")
        throw Error.invalidFile
    }
    let contextSize = contextSize(params: params, wType: wType)
    logger.info("Context size: \(contextSize / (1024.0 * 1024.0)) MB")
    return try loadModelData(
        stream: stream,
        params: params,
        vocab: vocab,
        contextSize: Int(contextSize),
        wType: wType
    )
}

internal func tokenize(
    string: String,
    vocab: Vocab
) throws -> [Int] {
    let pattern = try Regex(#"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"#)
    let ranges = string.ranges(of: pattern)
    let words = ranges.map { String(string[$0]) }
    var result = [Int]()
    for word in words {
        var leftIndex = word.startIndex
        while leftIndex < word.endIndex {
            var rightIndex = word.endIndex
            while leftIndex < rightIndex {
                let range = leftIndex ..< rightIndex
                if let token = vocab.tokenToId[String(word[range])] {
                    result.append(token)
                    leftIndex = string.index(before: rightIndex)
                    break
                }
                rightIndex = string.index(before: rightIndex)
            }
            leftIndex = string.index(after: leftIndex)
        }
    }
    return result
}

internal func sampleTopKTopP(
    vocab: Vocab,
    logits: [Float],
    topK: Int,
    topP: Double,
    temp: Double
) throws -> Int {
    let nLogits = vocab.idToToken.count
    var logitsId = [(Double, Int)]()
    let scale = 1.0 / temp
    for i in 0 ..< nLogits {
        logitsId.append((Double(logits[i]) * scale, i))
    }
    logitsId.sort { $0.0 > $1.0 }
    logitsId = Array(logitsId.prefix(topK))
    var maxl = -Double.infinity
    for (l, _) in logitsId {
        maxl = max(maxl, l)
    }

    var probs = [Double]()
    probs.reserveCapacity(logitsId.count)

    var sum = 0.0
    for (l, _) in logitsId {
        let p = exp(l - maxl)
        probs.append(p)
        sum += p
    }
    for index in probs.indices {
        probs[index] /= sum
    }

    if topP < 1.0 {
        var cumsum = 0.0
        for i in 0 ..< topK {
            cumsum += probs[i]
            if cumsum >= topP {
                probs = probs.resized(to: i + 1, repeating: 0.0)
                logitsId = logitsId.resized(to: i + 1, repeating: (0.0, 0))
                break
            }
        }
        cumsum = 1.0 / cumsum
        for index in probs.indices {
            probs[index] *= cumsum
        }
    }
    let index = Int.discrete_distribution(probs)[]
    return logitsId[index].1
}

@discardableResult
internal func predict(
    model: Model,
    numberOfThreads: Int,
    nPast: Int,
    inputEmbeddings: [Int],
    memoryPerToken: inout Int
) throws -> [Float] {
    let N = inputEmbeddings.count
    var bufferSize = 256 * 1024 * 1024
    if memoryPerToken > 0, memoryPerToken * N > bufferSize {
        bufferSize = Int(1.1 * Float(memoryPerToken * N))
        logger.info("Reallocating buffer to \(bufferSize) bytes)")
    }
    let context = try Context(memorySize: bufferSize, noAlloc: false)
    var graph = Graph(numberOfThreads: numberOfThreads)

    let embd = try context.tensor(type: .i32, shape: [N])
    embd.copy(from: inputEmbeddings.map { Int32($0) })

    let position = try context.tensor(type: .i32, shape: [N])
    let positionData = (0 ..< N).map { Int32(nPast + $0) }
    position.copy(from: positionData)

    var inpL = try context.add(
        context.rows(model.wte, embd),
        context.rows(model.wpe, position)
    )

    for index in 0 ..< model.params.nLayer {
        // norm
        var cur = try context.norm(inpL)
        // cur = ln_1_g*cur + ln_1_b
        cur = try context.add(
            context.mul(
                context.repeat(model.layers[index].ln_1_g, cur),
                cur
            ),
            context.repeat(model.layers[index].ln_1_b, cur)
        )

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        cur = try context.matMul(model.layers[index].c_attn_attn_w, cur)
        cur = try context.add(
            context.repeat(model.layers[index].c_attn_attn_b, cur),
            cur
        )

        // self-attention
        let qcur = try context.view2d(
            cur,
            model.params.nEmbd,
            N,
            cur.numberOfBytes(at: 1),
            0 * MemoryLayout<Float>.size * model.params.nEmbd
        )

        let kcur = try context.view2d(
            cur,
            model.params.nEmbd,
            N,
            cur.numberOfBytes(at: 1),
            1 * MemoryLayout<Float>.size * model.params.nEmbd
        )

        let vcur = try context.view2d(
            cur,
            model.params.nEmbd,
            N,
            cur.numberOfBytes(at: 1),
            2 * MemoryLayout<Float>.size * model.params.nEmbd
        )

        // store key and value to memory
        if N >= 1 {
            let k = try context.view1d(
                model.memory_k,
                N * model.params.nEmbd,
                model.memory_k.elementSize * model.params.nEmbd * (index * model.params.nCtx + nPast)
            )
            let v = try context.view1d(
                model.memory_v,
                N * model.params.nEmbd,
                model.memory_v.elementSize * model.params.nEmbd * (index * model.params.nCtx + nPast)
            )

            try graph.buildForwardExpand(for: context.copy(kcur, k))
            try graph.buildForwardExpand(for: context.copy(vcur, v))
        }

        // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
        // [64, N, 12]
        let q = try context.permute(
            context.copy(
                qcur,
                context.tensor(
                    type: .f32,
                    shape: [model.params.nEmbd / model.params.nHead, model.params.nHead, N]
                )
            ),
            0, 2, 1, 3
        )

        // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
        // [64, n_past + N, 12]
        let k = try context.permute(
            context.reshape3d(
                context.view1d(
                    model.memory_k,
                    (nPast + N) * model.params.nEmbd,
                    index * model.params.nCtx * model.memory_k.elementSize * model.params.nEmbd
                ),
                model.params.nEmbd / model.params.nHead,
                model.params.nHead,
                nPast + N
            ),
            0, 2, 1, 3
        )

        // K * Q
        // [n_past + N, N, 12]
        let kq = try context.matMul(k, q)

        // KQ_scaled = KQ / sqrt(n_embd/n_head)
        // [n_past + N, N, 12]
        let kqScaled = try context.scaleInplace(
            kq,
            context.f32(1.0 / sqrt(Float(model.params.nEmbd) / Float(model.params.nHead)))
        )

        // KQ_masked = mask_past(KQ_scaled)
        // [n_past + N, N, 12]
        let kqMasked = try context.diagMaskInfInplace(kqScaled, nPast)

        // KQ = soft_max(KQ_masked)
        // [n_past + N, N, 12]
        let kqSoftMax = try context.softmaxInplace(kqMasked)

        // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
        // [n_past + N, 64, 12]
        let vTrans = try context.copy(
            context.permute(
                context.reshape3d(
                    context.view1d(
                        model.memory_v,
                        (nPast + N) * model.params.nEmbd,
                        index * model.params.nCtx * model.memory_v.elementSize * model.params.nEmbd
                    ),
                    model.params.nEmbd / model.params.nHead,
                    model.params.nHead,
                    nPast + N
                ),
                1, 2, 0, 3
            ),
            context.tensor(
                type: model.memory_v.type,
                shape: [nPast + N, model.params.nEmbd / model.params.nHead, model.params.nHead]
            )
        )

        // KQV = transpose(V) * KQ_soft_max
        // [64, N, 12]
        let kqv = try context.matMul(vTrans, kqSoftMax)

        // KQV_merged = KQV.permute(0, 2, 1, 3)
        // [64, 12, N]
        let kqvMerged = try context.permute(kqv, 0, 2, 1, 3)

        // cur = KQV_merged.contiguous().view(n_embd, N)
        // [768, N]
        cur = try context.copy(
            kqvMerged,
            context.tensor(
                type: .f32,
                shape: [model.params.nEmbd, N]
            )
        )

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        cur = try context.matMul(
            model.layers[index].c_attn_proj_w,
            cur
        )
        cur = try context.add(
            context.repeat(
                model.layers[index].c_attn_proj_b,
                cur
            ),
            cur
        )

        // add the input
        cur = try context.add(cur, inpL)

        let inpff = cur

        // feed-forward network

        // norm
        cur = try context.norm(inpff)
        // cur = ln_2_g*cur + ln_2_b
        // [ 768, N]
        cur = try context.add(
            context.mul(
                context.repeat(
                    model.layers[index].ln_2_g,
                    cur
                ),
                cur
            ),
            context.repeat(
                model.layers[index].ln_2_b,
                cur
            )
        )
        // fully connected
        // [3072, 768] - model.layers[il].c_mlp_fc_w
        // [3072,   1] - model.layers[il].c_mlp_fc_b
        // [ 768,   N] - cur (in)
        // [3072,   N] - cur (out)
        //
        // cur = fc_w*cur + fc_b
        // [3072, N]
        cur = try context.matMul(
            model.layers[index].c_mlp_fc_w,
            cur
        )
        cur = try context.add(
            context.repeat(
                model.layers[index].c_mlp_fc_b,
                cur
            ),
            cur
        )

        // GELU activation
        // [3072, N]
        cur = try context.gelu(cur)

        // projection
        // [ 768, 3072] - model.layers[il].c_mlp_proj_w
        // [ 768,    1] - model.layers[il].c_mlp_proj_b
        // [3072,    N] - cur (in)
        // [ 768,    N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        cur = try context.matMul(
            model.layers[index].c_mlp_proj_w,
            cur
        )
        cur = try context.add(
            context.repeat(
                model.layers[index].c_mlp_proj_b,
                cur
            ),
            cur
        )

        // input for next layer
        inpL = try context.add(cur, inpff)
    }

    // norm
    // [ 768, N]
    inpL = try context.norm(inpL)
    // inpL = ln_f_g*inpL + ln_f_b
    // [ 768, N]
    inpL = try context.add(
        context.mul(
            context.repeat(
                model.ln_f_g,
                inpL
            ),
            inpL
        ),
        context.repeat(
            model.ln_f_b,
            inpL
        )
    )
    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = try context.matMul(
        model.lm_head,
        inpL
    )
    // run the computation
    graph.buildForwardExpand(for: inpL)
    context.compute(graph: &graph)

    if memoryPerToken == 0 {
        memoryPerToken = context.usedMemory / N
    }
    let ptr = inpL.data.assumingMemoryBound(to: Float.self) + model.params.nVocab * (N - 1)
    return Array(UnsafeBufferPointer(start: ptr, count: model.params.nVocab))
}
