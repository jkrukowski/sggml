import Foundation
import ggml

internal enum Error: Swift.Error {
    case invalidModelPath
    case invalidFile
    case failedToAllocateMemory
}

internal enum Constants {
    internal static let nInput = 784
    internal static let nHidden = 500
    internal static let nClasses = 10
    internal static let imageCount = 1000
}

internal struct Params {
    internal var nInput: Int32
    internal var nHidden: Int32
    internal var nClasses: Int32
}

internal struct Layer {
    internal var weight: UnsafeMutablePointer<ggml_tensor>
    internal var bias: UnsafeMutablePointer<ggml_tensor>
}

internal struct Model: CustomDebugStringConvertible {
    internal var layer1: Layer
    internal var layer2: Layer
    internal var params: Params
    internal var context: OpaquePointer

    internal var debugDescription: String {
        "<Mnist.Model: nInput: \(params.nInput), nHidden: \(params.nHidden), nClasses: \(params.nClasses)>"
    }
}
