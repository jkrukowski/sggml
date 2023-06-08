import ggml

public struct ScratchBuffer {
    internal let ggmlScratch: ggml_scratch

    public init(offset: Int, size: Int) {
        let data = UnsafeMutableRawPointer.allocate(
            byteCount: size,
            alignment: MemoryLayout<UInt8>.alignment
        )
        ggmlScratch = ggml_scratch(offs: offset, size: size, data: data)
    }
}
