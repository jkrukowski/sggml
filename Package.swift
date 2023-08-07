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
        .executable(
            name: "bert-spam",
            targets: ["BertSpam"]
        ),
        .library(
            name: "Tokenizer",
            targets: ["Tokenizer"]
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
            name: "ggml",
            cSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32", "-Ofast"], .when(configuration: .release)),
                .unsafeFlags(["-Wno-shorten-64-to-32"], .when(configuration: .debug)),
                .define("GGML_USE_ACCELERATE", .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
        .target(
            name: "Tokenizer",
            dependencies: [
            ],
            resources: [
                .copy("vocab.txt")
            ]
        ),
        .target(
            name: "Sggml",
            dependencies: [
                "ggml"
            ]
        ),
        .executableTarget(
            name: "BertSpam",
            dependencies: [
                "Sggml",
                "Utils",
                "Tokenizer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log")
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
