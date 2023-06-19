import ArgumentParser
import Foundation
import ggml
import Logging

private let logger = Logger(label: "sggml")

@main
internal struct Mnist: ParsableCommand {
    @Argument(help: "Path to MNIST model", transform: URL.init(fileURLWithPath:))
    internal var modelPath: URL

    @Argument(help: "Path to images file", transform: URL.init(fileURLWithPath:))
    internal var imagesPath: URL

    internal mutating func run() throws {
        let numberOfThreads = ProcessInfo.processInfo.activeProcessorCount
        logger.info("Using \(numberOfThreads) threads")
        let model = try readModel(at: modelPath)
        logger.info("loaded model \(model)")
        let image = try readImage(at: imagesPath, index: Int.random(in: 0 ..< Constants.imageCount))
        print(image: image)
        let predicted = try predict(image: image, model: model, numberOfThreads: numberOfThreads)
        logger.info("predicted number = \(predicted)")
    }
}
