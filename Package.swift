// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Sggml",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "Sggml",
            targets: ["Sggml"]
        ),
        .executable(
            name: "Mnist", 
            targets: ["Mnist"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.4.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
        .package(url: "https://github.com/tetsuok/swift-filestream", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "ggml",
            path: "Sources/ggml"
        ),
        .target(
            name: "Sggml",
            dependencies: [
                .target(name: "ggml"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/Sggml"
        ),
        .executableTarget(
            name: "Mnist",
            dependencies: [
                "Sggml",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "FileStream", package: "swift-filestream")
            ],
            path: "Sources/Mnist",
            resources: [
                .copy("models/mnist_model.state_dict"),
                .copy("models/t10k-images.idx3-ubyte"),
                .copy("models/ggml-model-f32.bin")
            ]
        ),
        .testTarget(
            name: "SggmlTests",
            dependencies: ["Sggml"]
        )
    ]
)
