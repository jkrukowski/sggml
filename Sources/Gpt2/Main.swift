import ArgumentParser
import Foundation
import Logging
import Sggml
import Utils

private let logger = Logger(label: "sggml")

@main
internal struct Gpt2: ParsableCommand {
    @Argument(help: "Path to GPT2 model", transform: URL.init(fileURLWithPath:))
    internal var modelPath: URL

    @Argument(help: "Prompt to generate text from")
    internal var prompt: String

    @Argument(help: "Number of tokens to predict")
    internal var predictCount: Int = 200

    @Argument(help: "Batch size for prompt processing")
    internal var promptBatchSize: Int = 8

    internal mutating func run() throws {
        let numberOfThreads = ProcessInfo.processInfo.activeProcessorCount
        logger.info("Using \(numberOfThreads) threads")
        let model = try readModel(at: modelPath)
        let inputEmbeddings = try tokenize(string: prompt, vocab: model.vocab)
        let predictCount = min(predictCount, model.params.nCtx - inputEmbeddings.count)

        // determine the required inference memory per token:
        var memoryPerToken = 0
        try predict(
            model: model,
            numberOfThreads: numberOfThreads,
            nPast: 0,
            inputEmbeddings: [0, 1, 2, 4],
            memoryPerToken: &memoryPerToken
        )

        let chunkedInputEmbeddings = inputEmbeddings.chunked(into: promptBatchSize)
        for (index, input) in chunkedInputEmbeddings.dropLast().enumerated() {
            try predict(
                model: model,
                numberOfThreads: numberOfThreads,
                nPast: index * input.count,
                inputEmbeddings: input,
                memoryPerToken: &memoryPerToken
            )
            for tokenId in input {
                print(model.vocab.idToToken[tokenId, default: ""], terminator: "")
            }
        }
        guard let lastChunkedInput = chunkedInputEmbeddings.last else {
            return
        }
        var input = lastChunkedInput
        var nPast = (chunkedInputEmbeddings.count - 1) * promptBatchSize
        for _ in 0 ... predictCount {
            for tokenId in input {
                print(model.vocab.idToToken[tokenId, default: ""], terminator: "")
            }
            let embeddings = try predict(
                model: model,
                numberOfThreads: numberOfThreads,
                nPast: nPast,
                inputEmbeddings: input,
                memoryPerToken: &memoryPerToken
            )
            let index = try sampleTopKTopP(
                vocab: model.vocab,
                logits: embeddings,
                topK: 40,
                topP: 0.9,
                temp: 0.9
            )
            if index == 50256 {
                print("")
                logger.info("Done: end of text token found")
                break
            }
            nPast += input.count
            input = [index]
        }
    }
}
