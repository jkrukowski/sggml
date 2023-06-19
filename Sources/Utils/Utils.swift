import Foundation

extension InputStream {
    public func read<T: BinaryInteger>() -> T {
        var data = [UInt8](repeating: 0, count: MemoryLayout<T>.size)
        read(&data, maxLength: data.count)
        return data.withUnsafeBytes { pointer in
            pointer.load(as: T.self)
        }
    }
}

public func verifyMagic(stream: InputStream) -> Bool {
    let number: Int32 = stream.read()
    return number == 0x67676D6C
}

public enum Error: Swift.Error {
    case invalidModelPath
    case invalidFile
    case failedToAllocateMemory
}

extension Array {
    public func resized(to newLength: Int, repeating element: Element) -> Self {
        if newLength > count {
            return self + [Element](repeating: element, count: newLength - count)
        } else if newLength < count {
            return Array(self[0 ..< newLength])
        } else {
            return self
        }
    }

    public func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}
