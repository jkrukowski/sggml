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
            name: "mnist",
            targets: ["Mnist"]
        ),
        .executable(
            name: "gpt2",
            targets: ["Gpt2"]
        ),
        .library(
            name: "Sggml",
            targets: ["Sggml"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.4.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
        .package(url: "https://github.com/jph00/BaseMath.git", branch: "master")
    ],
    targets: [
        .target(
            name: "Utils"
        ),
        .target(
            name: "ggml"
        ),
        .target(
            name: "Sggml",
            dependencies: [
                "ggml"
            ]
        ),
        .executableTarget(
            name: "Mnist",
            dependencies: [
                "Sggml",
                "Utils",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        .executableTarget(
            name: "Gpt2",
            dependencies: [
                "Sggml",
                "Utils",
                "BaseMath",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log")
            ]
        )
    ]
)
