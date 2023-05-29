import Foundation
import ggml
import Logging

private let logger = Logger(label: "sggml")

private func read<T: BinaryInteger>(from stream: InputStream) -> T {
    var data = [UInt8](repeating: 0, count: MemoryLayout<T>.size)
    stream.read(&data, maxLength: data.count)
    return data.withUnsafeBytes { pointer in
        pointer.load(as: T.self)
    }
}

private func readLayer1(
    stream: InputStream,
    context: OpaquePointer,
    params: inout Params
) throws -> Layer {
    let nDims: Int32 = read(from: stream)

    var neWeight: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neWeight[Int(index)] = read(from: stream)
    }
    let nInput = neWeight[0]
    let nHidden = neWeight[1]
    params.nInput = nInput
    params.nHidden = nHidden

    guard let weight = ggml_new_tensor_2d(context, GGML_TYPE_F32, Int64(nInput), Int64(nHidden)) else {
        throw Error.failedToAllocateMemory
    }
    stream.read(weight.pointee.data, maxLength: ggml_nbytes(weight))
    ggml_set_name(weight, "fc1_weight")

    var neBias: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neBias[Int(index)] = read(from: stream)
    }

    guard let bias = ggml_new_tensor_1d(context, GGML_TYPE_F32, Int64(nHidden)) else {
        throw Error.failedToAllocateMemory
    }
    stream.read(bias.pointee.data, maxLength: ggml_nbytes(bias))
    ggml_set_name(bias, "fc1_bias")
    return Layer(weight: weight, bias: bias)
}

private func readLayer2(
    stream: InputStream,
    context: OpaquePointer,
    params: inout Params
) throws -> Layer {
    let nDims: Int32 = read(from: stream)

    var neWeight: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neWeight[Int(index)] = read(from: stream)
    }
    let nClasses = neWeight[1]
    params.nClasses = nClasses

    guard let weight = ggml_new_tensor_2d(context, GGML_TYPE_F32, Int64(params.nHidden), Int64(nClasses)) else {
        throw Error.failedToAllocateMemory
    }
    stream.read(weight.pointee.data, maxLength: ggml_nbytes(weight))
    ggml_set_name(weight, "fc2_weight")

    var neBias: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neBias[Int(index)] = read(from: stream)
    }

    guard let bias = ggml_new_tensor_1d(context, GGML_TYPE_F32, Int64(nClasses)) else {
        throw Error.failedToAllocateMemory
    }
    stream.read(bias.pointee.data, maxLength: ggml_nbytes(bias))
    ggml_set_name(bias, "fc2_bias")
    return Layer(weight: weight, bias: bias)
}

private func verifyMagic(stream: InputStream) -> Bool {
    let number: Int32 = read(from: stream)
    return number == 0x67676D6C
}

internal func readModel(at modelPath: URL) throws -> Model {
    guard let stream = InputStream(fileAtPath: modelPath.path()) else {
        throw Error.invalidFile
    }
    stream.open()
    defer { stream.close() }

    if !verifyMagic(stream: stream) {
        throw Error.invalidFile
    }

    var contextSize = 0
    contextSize += Constants.nInput * Constants.nHidden * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc1 weight
    contextSize += Constants.nHidden * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc1 bias
    contextSize += Constants.nHidden * Constants.nClasses * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc2 weight
    contextSize += Constants.nClasses * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc2 bias

    logger.info("ggml ctx size = \(Float(contextSize) / (1024.0 * 1024.0)) MB")

    let contextParams = ggml_init_params(
        mem_size: contextSize + 1024 * 1024,
        mem_buffer: nil,
        no_alloc: false
    )

    guard let context = ggml_init(contextParams) else {
        throw Error.failedToAllocateMemory
    }

    var params = Params(nInput: 0, nHidden: 0, nClasses: 0)
    let layer1 = try readLayer1(stream: stream, context: context, params: &params)
    let layer2 = try readLayer2(stream: stream, context: context, params: &params)
    return Model(layer1: layer1, layer2: layer2, params: params, context: context)
}

internal func predict(
    image: [Float],
    model: Model,
    numberOfThreads: Int = 1
) throws -> Int {
    let bufferSize = Int(model.params.nInput) * 4 * MemoryLayout<Float>.alignment
    let buffer = UnsafeMutableRawPointer.allocate(
        byteCount: bufferSize,
        alignment: MemoryLayout<Float>.alignment
    )

    let contextParams = ggml_init_params(
        mem_size: bufferSize,
        mem_buffer: buffer,
        no_alloc: false
    )

    guard let context = ggml_init(contextParams) else {
        throw Error.failedToAllocateMemory
    }
    defer { ggml_free(context) }

    var graph = ggml_cgraph()
    graph.n_threads = Int32(numberOfThreads)

    guard let input = ggml_new_tensor_1d(context, GGML_TYPE_F32, Int64(model.params.nInput)) else {
        throw Error.failedToAllocateMemory
    }
    input.pointee.data.copyMemory(from: image, byteCount: ggml_nbytes(input))
    ggml_set_name(input, "input")

    guard let fc1 = ggml_add(context, ggml_mul_mat(context, model.layer1.weight, input), model.layer1.bias) else {
        throw Error.failedToAllocateMemory
    }
    guard let fc2 = ggml_add(context, ggml_mul_mat(context, model.layer2.weight, ggml_relu(context, fc1)), model.layer2.bias) else {
        throw Error.failedToAllocateMemory
    }
    guard let probabilities = ggml_soft_max(context, fc2) else {
        throw Error.failedToAllocateMemory
    }
    ggml_set_name(probabilities, "probabilities")

    ggml_build_forward_expand(&graph, probabilities)
    ggml_graph_compute(context, &graph)

    guard let probabilitiesData = ggml_get_data_f32(probabilities) else {
        throw Error.failedToAllocateMemory
    }
    let prediction = Array(UnsafeBufferPointer(start: probabilitiesData, count: Int(model.params.nClasses)))
    var maxElement = prediction[0]
    var maxIndex = 0
    for (index, element) in prediction.enumerated() {
        if element > maxElement {
            maxElement = element
            maxIndex = index
        }
    }
    return maxIndex
}

internal func readImage(at url: URL, index: Int) throws -> [Float] {
    guard let stream = InputStream(fileAtPath: url.path()) else {
        throw Error.invalidFile
    }
    stream.open()
    defer { stream.close() }

    var headerData = [UInt8](repeating: 0, count: 16)
    stream.read(&headerData, maxLength: headerData.count)

    let bufferSize = 784
    var startIndex = 0
    let endIndex = index % Constants.imageCount
    logger.info("reading image \(endIndex)")
    while stream.hasBytesAvailable {
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        stream.read(buffer, maxLength: bufferSize)
        if startIndex == endIndex {
            return Array(UnsafeBufferPointer(start: buffer, count: bufferSize)).map { Float($0) }
        }
        startIndex += 1
    }
    throw Error.invalidFile
}

internal func print(image: [Float]) {
    for row in 0 ..< 28 {
        for col in 0 ..< 28 {
            let value = image[row * 28 + col]
            if value > 230 {
                print("*", terminator: "")
            } else {
                print("_", terminator: "")
            }
        }
        print("")
    }
}
