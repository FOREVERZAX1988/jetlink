import Foundation

/// ONNX element types the engine stages, with their sizes.
public enum ElementType: Int32, Sendable {
  case float = 1
  case uint8 = 2
  case int8 = 3
  case uint16 = 4
  case int16 = 5
  case int32 = 6
  case int64 = 7
  case bool = 9
  case float16 = 10
  case double = 11

  public var size: Int {
    switch self {
    case .uint8, .int8, .bool: 1
    case .uint16, .int16, .float16: 2
    case .float, .int32: 4
    case .int64, .double: 8
    }
  }

  public var name: String {
    switch self {
    case .float: "float32"
    case .uint8: "uint8"
    case .int8: "int8"
    case .uint16: "uint16"
    case .int16: "int16"
    case .int32: "int32"
    case .int64: "int64"
    case .bool: "bool"
    case .float16: "float16"
    case .double: "float64"
    }
  }
}

/// One input or output as a session declares it.
public struct TensorSpec: Sendable, Equatable {
  public let name: String
  public let type: ElementType
  public let shape: [Int]

  public var count: Int { shape.reduce(1, *) }
  public var byteCount: Int { count * type.size }
}
