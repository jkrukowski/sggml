import ArgumentParser
import FileStream
import Foundation
import ggml
import Logging

private let logger = Logger(label: "sggml")

@main
internal struct Mnist: ParsableCommand {
    internal init() {}

    internal mutating func run() throws {
        guard let modelPath = Bundle.module.url(forResource: "ggml-model-f32", withExtension: "bin") else {
            throw Error.invalidModelPath
        }
        guard let imagesPath = Bundle.module.url(forResource: "t10k-images", withExtension: "idx3-ubyte") else {
            throw Error.invalidModelPath
        }
        let model = try readModel(at: modelPath)
        defer { ggml_free(model.context) }
        logger.info("loaded model \(model)")
        let image = try readImage(at: imagesPath, index: Int.random(in: 0 ..< Constants.imageCount))
        print(image: image)
        let predicted = try predict(image: image, model: model)
        logger.info("predicted number = \(predicted)")
    }
}
