import Foundation
import Logging
import ArgumentParser
import FileStream
import ggml

internal let logger = Logger(label: "Sggml")

extension Mnist {
    internal enum Error: Swift.Error {
        case invalidModelPath
        case invalidFile
    }
}

extension Mnist {
    internal struct Constants {
        internal static let nInput = 784
        internal static let nHidden = 500
        internal static let nClasses = 10
    }
}

extension Mnist {
    internal struct Params {
        internal var nInput: Int32
        internal var nHidden: Int32
        internal var nClasses: Int32
    }
}

extension Mnist {
    internal struct Layer {
        internal var weight: UnsafeMutablePointer<ggml_tensor>
        internal var bias: UnsafeMutablePointer<ggml_tensor>
    }
}

extension Mnist {
    internal struct Model: CustomDebugStringConvertible {
        internal var layer1: Layer
        internal var layer2: Layer
        internal var params: Params
        internal var context: OpaquePointer

        internal var debugDescription: String {
            "<Mnist.Model: nInput: \(params.nInput), nHidden: \(params.nHidden), nClasses: \(params.nClasses)>"
        }
    }
}

private func readLayer1(
    stream: InputStream, 
    context: OpaquePointer,
    params: inout Mnist.Params
) throws -> Mnist.Layer {
    var nDimsData = [UInt8](repeating: 0, count: 4)
    stream.read(&nDimsData, maxLength: nDimsData.count)
    let nDims = nDimsData.withUnsafeBytes { pointer in
        return pointer.load(as: Int32.self)
    }

    var neWeight: [Int32] = [0, 0]
    for index in 0..<nDims {
        var neWeightData = [UInt8](repeating: 0, count: 4)
        stream.read(&neWeightData, maxLength: neWeightData.count)
        neWeight[Int(index)] = neWeightData.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        }
    }
    let nInput = neWeight[0]
    let nHidden = neWeight[1]
    params.nInput = nInput
    params.nHidden = nHidden

    guard let weight = ggml_new_tensor_2d(context, GGML_TYPE_F32, Int64(nInput), Int64(nHidden)) else {
        throw Mnist.Error.invalidFile
    }
    stream.read(weight.pointee.data, maxLength: ggml_nbytes(weight))
    ggml_set_name(weight, "fc1_weight")

    var neBias: [Int32] = [0, 0]
    for index in 0..<nDims {
        var neBiasData = [UInt8](repeating: 0, count: 4)
        stream.read(&neBiasData, maxLength: neBiasData.count)
        neBias[Int(index)] = neBiasData.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        }
    }

    guard let bias = ggml_new_tensor_1d(context, GGML_TYPE_F32, Int64(nHidden)) else {
        throw Mnist.Error.invalidFile
    }
    stream.read(bias.pointee.data, maxLength: ggml_nbytes(bias))
    ggml_set_name(bias, "fc1_bias")
    return Mnist.Layer(weight: weight, bias: bias)
}

private func readLayer2(
    stream: InputStream, 
    context: OpaquePointer,
    params: inout Mnist.Params
) throws -> Mnist.Layer {
    var nDimsData = [UInt8](repeating: 0, count: 4)
    stream.read(&nDimsData, maxLength: nDimsData.count)
    let nDims = nDimsData.withUnsafeBytes { pointer in
        return pointer.load(as: Int32.self)
    }

    var neWeight: [Int32] = [0, 0]
    for index in 0..<nDims {
        var neWeightData = [UInt8](repeating: 0, count: 4)
        stream.read(&neWeightData, maxLength: neWeightData.count)
        neWeight[Int(index)] = neWeightData.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        }
    }
    let nClasses = neWeight[1]
    params.nClasses = nClasses

    guard let weight = ggml_new_tensor_2d(context, GGML_TYPE_F32, Int64(params.nHidden), Int64(nClasses)) else {
        throw Mnist.Error.invalidFile
    }
    stream.read(weight.pointee.data, maxLength: ggml_nbytes(weight))
    ggml_set_name(weight, "fc2_weight")

    var neBias: [Int32] = [0, 0]
    for index in 0..<nDims {
        var neBiasData = [UInt8](repeating: 0, count: 4)
        stream.read(&neBiasData, maxLength: neBiasData.count)
        neBias[Int(index)] = neBiasData.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        }
    }

    guard let bias = ggml_new_tensor_1d(context, GGML_TYPE_F32, Int64(nClasses)) else {
        throw Mnist.Error.invalidFile
    } 
    stream.read(bias.pointee.data, maxLength: ggml_nbytes(bias))
    ggml_set_name(bias, "fc2_bias")
    return Mnist.Layer(weight: weight, bias: bias)
}

private func readModel() throws -> Mnist.Model {
    guard let modelPath = Bundle.module.url(forResource: "ggml-model-f32", withExtension: "bin") else {
        throw Mnist.Error.invalidModelPath
    }
    guard let stream = InputStream(fileAtPath: modelPath.path()) else {
        throw Mnist.Error.invalidFile
    }
    stream.open()
    defer { stream.close() }

    var numberData = [UInt8](repeating: 0, count: 4)
    stream.read(&numberData, maxLength: numberData.count)
    let number = numberData.withUnsafeBytes { pointer in
        return pointer.load(as: Int32.self)
    }

    if number != 0x67676d6c {
        throw Mnist.Error.invalidFile
    }

    var contextSize = 0
    contextSize += Mnist.Constants.nInput * Mnist.Constants.nHidden * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc1 weight
    contextSize += Mnist.Constants.nHidden * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc1 bias
    contextSize += Mnist.Constants.nHidden * Mnist.Constants.nClasses * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc2 weight
    contextSize += Mnist.Constants.nClasses * Int(ggml_type_sizef(GGML_TYPE_F32)) // fc2 bias

    logger.info("ggml ctx size = \(Float(contextSize)/(1024.0*1024.0)) MB")

    let contextParams = ggml_init_params(
        mem_size: contextSize + 1024*1024, 
        mem_buffer: nil, 
        no_alloc: false
    )

    guard let context = ggml_init(contextParams) else {
        throw Mnist.Error.invalidFile
    }

    var params = Mnist.Params(nInput: 0, nHidden: 0, nClasses: 0)
    let layer1 = try readLayer1(stream: stream, context: context, params: &params)
    let layer2 = try readLayer2(stream: stream, context: context, params: &params)
    return Mnist.Model(layer1: layer1, layer2: layer2, params: params, context: context)
}

@main
internal struct Mnist: ParsableCommand {
    internal init() {}

    internal mutating func run() throws {
        let model = try readModel()
        logger.info("ggml loaded model \(model)")
    }
}