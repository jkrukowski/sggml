import Foundation
import ggml
import Logging
import Sggml
import Utils

private let logger = Logger(label: "sggml")

private func readLayer1(
    stream: InputStream,
    context: Context,
    params: inout Params
) throws -> Layer {
    let nDims: Int32 = stream.read()

    var neWeight: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neWeight[Int(index)] = stream.read()
    }
    let nInput = Int(neWeight[0])
    let nHidden = Int(neWeight[1])
    params.nInput = nInput
    params.nHidden = nHidden

    var weight = try context.tensor(type: .f32, shape: [nInput, nHidden])
    stream.read(weight.data, maxLength: weight.byteCount)
    weight.name = "fc1_weight"

    var neBias: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neBias[Int(index)] = stream.read()
    }

    var bias = try context.tensor(type: .f32, shape: [nHidden])
    stream.read(bias.data, maxLength: bias.byteCount)
    bias.name = "fc1_bias"

    return Layer(weight: weight, bias: bias)
}

private func readLayer2(
    stream: InputStream,
    context: Context,
    params: inout Params
) throws -> Layer {
    let nDims: Int32 = stream.read()

    var neWeight: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neWeight[Int(index)] = stream.read()
    }
    let nClasses = Int(neWeight[1])
    params.nClasses = nClasses

    var weight = try context.tensor(type: .f32, shape: [params.nHidden, nClasses])
    stream.read(weight.data, maxLength: weight.byteCount)
    weight.name = "fc2_weight"

    var neBias: [Int32] = [0, 0]
    for index in 0 ..< nDims {
        neBias[Int(index)] = stream.read()
    }

    var bias = try context.tensor(type: .f32, shape: [nClasses])
    stream.read(bias.data, maxLength: bias.byteCount)
    bias.name = "fc2_bias"

    return Layer(weight: weight, bias: bias)
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

    let f32TypeSize = Int(TensorType.f32.typeSize)
    var contextSize = 0
    contextSize += Constants.nInput * Constants.nHidden * f32TypeSize // fc1 weight
    contextSize += Constants.nHidden * f32TypeSize // fc1 bias
    contextSize += Constants.nHidden * Constants.nClasses * f32TypeSize // fc2 weight
    contextSize += Constants.nClasses * f32TypeSize // fc2 bias

    logger.info("ggml ctx size = \(Float(contextSize) / (1024.0 * 1024.0)) MB")

    let context = try Context(memorySize: contextSize + 1024 * 1024, noAlloc: false)

    var params = Params(nInput: 0, nHidden: 0, nClasses: 0)
    let layer1 = try readLayer1(stream: stream, context: context, params: &params)
    let layer2 = try readLayer2(stream: stream, context: context, params: &params)
    return Model(layer1: layer1, layer2: layer2, params: params, context: context)
}

internal func predict(
    image: [Float],
    model: Model,
    numberOfThreads: Int
) throws -> Int {
    let bufferSize = Int(model.params.nInput) * 4 * MemoryLayout<Float>.alignment
    let context = try Context(memorySize: bufferSize, noAlloc: false)

    var input = try context.tensor(type: .f32, shape: [model.params.nInput])
    input.copy(from: image)
    input.name = "input"

    let fc1 = try context.add(context.matMul(model.layer1.weight, input), model.layer1.bias)
    let fc2 = try context.add(context.matMul(model.layer2.weight, context.relu(fc1)), model.layer2.bias)
    var probabilities = try context.softmax(fc2)
    probabilities.name = "probabilities"

    var graph = Graph(numberOfThreads: numberOfThreads)
    graph.buildForwardExpand(for: probabilities)
    context.compute(graph: &graph)

    let prediction: [Float] = try probabilities.data(count: model.params.nClasses)
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
