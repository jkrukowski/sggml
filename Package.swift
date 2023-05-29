// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Sggml",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "Mnist",
            targets: ["Mnist"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.4.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0")
    ],
    targets: [
        .target(
            name: "ggml",
            path: "Sources/ggml"
        ),
        .executableTarget(
            name: "Mnist",
            dependencies: [
                "ggml",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/Mnist",
            resources: [
                .copy("models/t10k-images.idx3-ubyte"),
                .copy("models/ggml-model-f32.bin")
            ]
        )
    ]
)
