import ArgumentParser
import Foundation
import ggml
import Logging
import Tokenizer
import Utils

private let logger = Logger(label: "bert-spam")

@main
internal struct Bert: ParsableCommand {
    @Argument(help: "Path to model", transform: URL.init(fileURLWithPath:))
    internal var modelPath: URL

    @Argument(help: "Text")
    internal var text: String

    internal mutating func run() throws {
        let numberOfThreads = ProcessInfo.processInfo.activeProcessorCount
        logger.info("Using \(numberOfThreads) threads")
        let tokenizer = BertTokenizer()
        let model = try readModel(at: modelPath)
        let memoryPerTokenResult = try predict(
            model: model,
            numberOfThreads: numberOfThreads,
            config: .memoryPerToken
        )
        guard let memoryPerToken = memoryPerTokenResult.memoryPerToken else {
            logger.error("Failed to get memory per token")
            throw Error.failedToProcess
        }
        let inputEmbeddings = tokenizer.tokenizeToIds(text: text)
        logger.info("Memory per token: \(memoryPerToken), token count: \(inputEmbeddings.count)")
        let predictionResult = try predict(
            model: model,
            numberOfThreads: numberOfThreads,
            config: .prediction(
                inputEmbeddings: inputEmbeddings,
                memoryPerToken: memoryPerToken
            )
        )
        guard let prediction = predictionResult.prediction else {
            logger.error("Failed to get prediction")
            throw Error.failedToProcess
        }
        logger.info("Result: \(prediction)")
    }
}
