/// An identifier of a given abstract value with the value's type attached
///
/// - Parameter Value: the type of value this ID refers to.
/// - Parameter PerTypeID: a type that, given `Value`, identifies a given
///   logical value of that type.
public struct TypedID<Value, PerTypeID: Equatable> {
  /// A specifier of which logical value of type `value` is being identified.
  let perTypeID: PerTypeID

  /// Creates an instance indicating the given logical value of type `Value`.
  init(_ perTypeID: PerTypeID) { self.perTypeID = perTypeID }
}

public typealias SimpleTypedID<Value> = TypedID<Value, Int>

public typealias VariableAssignments = PackedStorage

// MARK: - Storage without any operations.

/// An existential type for instances of `HomogeneousStorage` that is agnostic to
/// the element type of the underlying array.
///
/// Any given model is associated with a particular element type.
protocol AnyHomogeneousStorage {
  /// Appends the instance of the concrete element type whose address is `p`,
  /// returning the index of the appended element.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int
  
  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  mutating func mutableAddress(ofElement n: Int) -> UnsafeMutableRawPointer

  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  func address(ofElement i: Int) -> UnsafeRawPointer
  
  /// Returns a mutable buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var mutableBuffer: UnsafeMutableRawBufferPointer { mutating get }

  /// Returns a buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var buffer: UnsafeRawBufferPointer { get }

  // MARK: - Factor methods.

  /// Returns the total error of all the factors.
  func error(at values: PackedStorage) -> Double

  /// Accumulates the error vectors of all the factors to `out`.
  /// Precondition: `out` has entries for the error vectors of all the elements.
  func errorVector(at values: PackedStorage, _ out: inout PackedStorage)

  /// Computes linear approximations of the elements at `values` and writes the approximations to
  /// `out`.
  func linearize(at values: PackedStorage, _ out: inout FactorGraph)

  /// Applies all the elements' forward linear maps to `x` and accumulates the results to `y`.
  /// Precondition: `y` has an entry at all the elements' outputs.
  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage)

  /// Applies all the elements' tranpose linear maps to `y` and accumulates the results to `x`.
  /// Precondition: `x` has an entry at all the elements' inputs.
  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage)

  // MARK: - Differentiable methods.

  mutating func move(along direction: AnyHomogeneousStorage)

  var zeroTangentVector: (ObjectIdentifier, AnyHomogeneousStorage) { get }

  // MARK: - EuclideanVectorSpace methods.

  func scaled(by scalar: Double) -> AnyHomogeneousStorage

  func plus (_ other: AnyHomogeneousStorage) -> AnyHomogeneousStorage

  var squaredNorm: Double { get }
}

extension AnyHomogeneousStorage {
  func error(at values: PackedStorage) -> Double {
    fatalError("unsupported")
  }

  func errorVector(at values: PackedStorage, _ out: inout PackedStorage) {
    fatalError("unsupported")
  }

  func linearize(at values: PackedStorage, _ out: inout FactorGraph) {
    fatalError("unsupported")
  }

  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage) {
    fatalError("unsupported")
  }

  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage) {
    fatalError("unsupported")
  }

  mutating func move(along direction: AnyHomogeneousStorage) {
    fatalError("unsupported")
  }

  var zeroTangentVector: (ObjectIdentifier, AnyHomogeneousStorage) {
    fatalError("unsupported")
  }

  func scaled(by scalar: Double) -> AnyHomogeneousStorage {
    fatalError("unsupported")
  }

  func plus (_ other: AnyHomogeneousStorage) -> AnyHomogeneousStorage {
    fatalError("unsupported")
  }

  var squaredNorm: Double {
    fatalError("unsupported")
  }
}

/// A contiguous buffer of `T` instances that can be handled as an existential.
struct HomogeneousStorage<T>: AnyHomogeneousStorage {
  /// The actual storage for `T` instances.
  var storage: [T] = []
  
  /// Appends the instance of the concrete element type whose address is `p`,
  /// returning the index of the appended element.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    storage.append(p.assumingMemoryBound(to: T.self).pointee)
    return storage.count - 1
  }
  
  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    withUnsafeMutablePointer(to: &storage[i]) { .init($0) }
  }

  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  func address(ofElement i: Int) -> UnsafeRawPointer {
    precondition(i >= 0 && i < storage.count, "index out of range")
    return storage.withUnsafeBufferPointer { .init($0.baseAddress! + i) }
  }

  /// Returns a mutable buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      .init(storage.withUnsafeMutableBufferPointer { $0 })
    }
  }
  
  /// Returns a buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var buffer: UnsafeRawBufferPointer {
    get {
      .init(storage.withUnsafeBufferPointer { $0 })
    }
  }
}

struct HomogeneousStorageFactor<T: AnyFactor>: AnyHomogeneousStorage {
  var base: HomogeneousStorage<T> = HomogeneousStorage()

  // Delegate implementation to `base`.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    base.appendValue(at: p)
  }
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    base.mutableAddress(ofElement: i)
  }
  func address(ofElement i: Int) -> UnsafeRawPointer {
    base.address(ofElement: i)
  }
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      base.mutableBuffer
    }
  }
  var buffer: UnsafeRawBufferPointer {
    get {
      base.buffer
    }
  }

  /// Returns the total error of all the factors.
  func error(at values: PackedStorage) -> Double {
    return T.InputIDs.withFastAssignments(values) { fastAssignments in
      return base.storage.reduce(0) { error, factor in error + factor.error(at: fastAssignments) }
    }
  }

  /// Accumulates the error vectors of all the factors to `out`.
  /// Precondition: `out` has entries for the error vectors of all the elements.
  func errorVector(
    at values: PackedStorage,
    _ out: inout PackedStorage
  ) {
    let factorsBuffer = base.buffer.bindMemory(to: T.self)
    let resultBuffer = out.homogeneousStorage[ObjectIdentifier(T.self)]!
      .mutableBuffer.bindMemory(to: T.ErrorVector.self)
    T.InputIDs.withFastAssignments(values) { fastAssignments in
      for index in 0..<factorsBuffer.count {
        let factor = factorsBuffer[index]
        resultBuffer[index] += factor.errorVector(at: fastAssignments)
      }
    }
  }

  /// Computes linear approximations of the elements at `values` and writes the approximations to
  /// `out`.
  @_specialize(where T == MyPriorFactor)
  @_specialize(where T == MyBetweenFactor)
  func linearize(
    at values: PackedStorage,
    _ out: inout FactorGraph
  ) {
    T.InputIDs.withFastAssignments(values) { fastAssignments in
      for factor in base.storage {
        out += factor.linearized(at: fastAssignments)
      }
    }
  }

  /// Applies all the elements' forward linear maps to `x` and accumulates the results to `y`.
  /// Precondition: `y` has an entry at all the elements' outputs.
  @_specialize(where T == JacobianFactor1<Matrix3x3>)
  @_specialize(where T == JacobianFactor2<Matrix3x3, Matrix3x3>)
  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage) {
    let factorsBuffer = base.buffer.bindMemory(to: T.self)
    let resultBuffer = y.homogeneousStorage[ObjectIdentifier(T.self)]!
      .mutableBuffer.bindMemory(to: T.ErrorVector.self)
    T.InputIDs.withFastAssignments(x) { fastAssignments in
      for index in 0..<factorsBuffer.count {
        let factor = factorsBuffer[index]
        resultBuffer[index] += factor.applyLinearForward(fastAssignments)
      }
    }
  }

  /// Applies all the elements' tranpose linear maps to `y` and accumulates the results to `x`.
  /// Precondition: `x` has an entry at all the elements' inputs.
  @_specialize(where T == JacobianFactor1<Matrix3x3>)
  @_specialize(where T == JacobianFactor2<Matrix3x3, Matrix3x3>)
  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage) {
    let factorsBuffer = base.buffer.bindMemory(to: T.self)
    let yBuffer = y.homogeneousStorage[ObjectIdentifier(T.self)]!
      .buffer.bindMemory(to: T.ErrorVector.self)
    T.InputIDs.withMutableFastAssignments(&x) { fastAssignments in
      for index in 0..<factorsBuffer.count {
        let factor = factorsBuffer[index]
        factor.applyLinearTranspose(yBuffer[index], into: &fastAssignments)
      }
    }
  }
}

struct HomogeneousStorageDifferentiable<T: Differentiable>: AnyHomogeneousStorage where T.TangentVector: EuclideanVectorSpace {
  var base: HomogeneousStorage<T> = HomogeneousStorage()

  // Delegate implementation to `base`.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    base.appendValue(at: p)
  }
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    base.mutableAddress(ofElement: i)
  }
  func address(ofElement i: Int) -> UnsafeRawPointer {
    base.address(ofElement: i)
  }
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      base.mutableBuffer
    }
  }
  var buffer: UnsafeRawBufferPointer {
    get {
      base.buffer
    }
  }

  mutating func move(along direction: AnyHomogeneousStorage) {
    let buffer = direction.buffer.bindMemory(to: T.TangentVector.self)
    for index in base.storage.indices {
      base.storage[index].move(along: buffer[index])
    }
  }

  var zeroTangentVector: (ObjectIdentifier, AnyHomogeneousStorage) {
    let buffer = self.buffer.bindMemory(to: T.self)
    let result = Array<T.TangentVector>(unsafeUninitializedCapacity: buffer.count) {
      resultBuffer, initializedCount in
      for index in 0..<buffer.count {
        resultBuffer[index] = buffer[index].zeroTangentVector
      }
      initializedCount = buffer.count
    }
    return (
      ObjectIdentifier(T.TangentVector.self),
      HomogeneousStorageEuclideanVector<T.TangentVector>(base: HomogeneousStorage(storage: result))
    )
  }
}

struct HomogeneousStorageEuclideanVector<T: EuclideanVectorSpace>: AnyHomogeneousStorage {
  var base: HomogeneousStorage<T> = HomogeneousStorage()

  // Delegate implementation to `base`.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    base.appendValue(at: p)
  }
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    base.mutableAddress(ofElement: i)
  }
  func address(ofElement i: Int) -> UnsafeRawPointer {
    base.address(ofElement: i)
  }
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      base.mutableBuffer
    }
  }
  var buffer: UnsafeRawBufferPointer {
    get {
      base.buffer
    }
  }

  mutating func move(along direction: AnyHomogeneousStorage) {
    let buffer = direction.buffer.bindMemory(to: T.TangentVector.self)
    for index in base.storage.indices {
      base.storage[index].move(along: buffer[index])
    }
  }

  var zeroTangentVector: AnyHomogeneousStorage {
    let buffer = self.buffer.bindMemory(to: T.self)
    let result = Array<T>(unsafeUninitializedCapacity: buffer.count) {
      resultBuffer, initializedCount in
      for index in 0..<buffer.count {
        resultBuffer[index] = buffer[index].zeroTangentVector
      }
      initializedCount = buffer.count
    }
    return HomogeneousStorageEuclideanVector<T>(base: HomogeneousStorage(storage: result))
  }

  func scaled(by scalar: Double) -> AnyHomogeneousStorage {
    let buffer = self.buffer.bindMemory(to: T.self)
    let result = Array<T>(unsafeUninitializedCapacity: buffer.count) {
      resultBuffer, initializedCount in
      for index in 0..<buffer.count {
        resultBuffer[index] = buffer[index].scaled(by: scalar)
      }
      initializedCount = buffer.count
    }
    return HomogeneousStorageEuclideanVector<T>(base: HomogeneousStorage(storage: result))
  }

  func plus (_ other: AnyHomogeneousStorage) -> AnyHomogeneousStorage {
    let ourBuffer = self.buffer.bindMemory(to: T.self)
    let otherBuffer = other.buffer.bindMemory(to: T.self)
    let result = Array<T>(unsafeUninitializedCapacity: base.storage.count) {
      resultBuffer, initializedCount in
      for index in 0..<ourBuffer.count {
        resultBuffer[index] = ourBuffer[index] + otherBuffer[index]
      }
      initializedCount = ourBuffer.count
    }
    return HomogeneousStorageEuclideanVector<T>(base: HomogeneousStorage(storage: result))
  }

  var squaredNorm: Double {
    var result = Double(0)
    let buffer = self.buffer.bindMemory(to: T.self)
    for index in 0..<buffer.count {
      result += buffer[index].squaredNorm
    }
    return result
  }
}

/// Storage for instances of heterogeneous type in contiguous homogeneous
/// memory.
public struct PackedStorage {
  public init() {  }
  
  /// A mapping from type onto the homogeneous storage for that type.
  var homogeneousStorage: [ObjectIdentifier: AnyHomogeneousStorage] = [:]

  /// A mapping from type onto the tangent vector for that type.
  var tangentVector: [ObjectIdentifier: ObjectIdentifier] = [:]

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func store<T>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int> {
    .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorage<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func storeFactor<T: AnyFactor>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int> {
    .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorageFactor<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func storeDifferentiable<T: Differentiable>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int>
    where T.TangentVector: EuclideanVectorSpace
  {
    tangentVector[ObjectIdentifier(T.self)] = ObjectIdentifier(T.TangentVector.self)
    return .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorageDifferentiable<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func storeEuclideanVector<T: EuclideanVectorSpace>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int> {
    tangentVector[ObjectIdentifier(T.self)] = ObjectIdentifier(T.TangentVector.self)
    return .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorageEuclideanVector<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Traps with an error indicating that an attempt was made to access a stored
  /// variable of a type that is not represented in `self`.
  private static var noSuchType: AnyHomogeneousStorage {
    fatalError("No such stored variable type")
  }

  /// Accesses the stored value with the given ID.
  public subscript<T>(id: TypedID<T, Int>, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> T {
    _read {
      yield homogeneousStorage[
        tag, default: Self.noSuchType]
        .address(ofElement: id.perTypeID)
        .assumingMemoryBound(to: T.self).pointee
    }
    _modify {
      defer { _fixLifetime(self) }
      yield &homogeneousStorage[
        tag, default: Self.noSuchType
      ]
        .mutableAddress(ofElement: id.perTypeID)
        .assumingMemoryBound(to: T.self).pointee
    }
  }

  // MARK: - Factor methods.

  /// Returns the total error of all the factors.
  func error(at values: PackedStorage) -> Double {
    var result = Double(0)
    for factors in homogeneousStorage.values {
      result += factors.error(at: values)
    }
    return result
  }

  /// Accumulates the error vectors of all the factors to `out`.
  /// Precondition: `out` has entries for the error vectors of all the elements.
  func errorVector(at values: PackedStorage, _ out: inout PackedStorage) {
    for factors in homogeneousStorage.values {
      factors.errorVector(at: values, &out)
    }
  }

  /// Computes linear approximations of the elements at `values` and writes the approximations to
  /// `out`.
  func linearize(at values: PackedStorage, _ out: inout FactorGraph) {
    for factors in homogeneousStorage.values {
      factors.linearize(at: values, &out)
    }
  }

  /// Applies all the elements' forward linear maps to `x` and accumulates the results to `y`.
  /// Precondition: `y` has an entry at all the elements' outputs.
  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage) {
    for factors in homogeneousStorage.values {
      factors.applyLinearForward(x, &y)
    }
  }

  /// Applies all the elements' tranpose linear maps to `y` and accumulates the results to `x`.
  /// Precondition: `x` has an entry at all the elements' inputs.
  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage) {
    for factors in homogeneousStorage.values {
      factors.applyLinearTranspose(y, &x)
    }
  }

  // MARK: - Differentiable methods.

  public mutating func move(along direction: PackedStorage) {
    for key in homogeneousStorage.keys {
      homogeneousStorage[key]!.move(along: direction.homogeneousStorage[tangentVector[key]!]!)
    }
  }

  public var zeroTangentVector: PackedStorage {
    var result = PackedStorage()
    for value in homogeneousStorage.values {
      let (objectID, zero) = value.zeroTangentVector
      result.homogeneousStorage[objectID] = zero
    }
    return result
  }

  // MARK: - EuclideanVectorSpace methods.

  public func scaled(by scalar: Double) -> PackedStorage {
    var result = PackedStorage()
    for (objectID, value) in homogeneousStorage {
      result.homogeneousStorage[objectID] = value.scaled(by: scalar)
    }
    return result
  }

  public static func + (_ lhs: PackedStorage, _ rhs: PackedStorage) -> PackedStorage {
    var result = PackedStorage()
    for (objectID, value) in lhs.homogeneousStorage {
      result.homogeneousStorage[objectID] = value.plus(rhs.homogeneousStorage[objectID]!)
    }
    return result
  }

  public var squaredNorm: Double {
    var result = Double(0)
    for value in homogeneousStorage.values {
      result += value.squaredNorm
    }
    return result
  }
}

extension PackedStorage: Equatable {
  public static func == (_ lhs: PackedStorage, _ rhs: PackedStorage) -> Bool {
    fatalError("unsupported")
  }
}

extension PackedStorage: EuclideanVectorSpace {
  public typealias TangentVector = PackedStorage
  public static func += (_ lhs: inout PackedStorage, _ rhs: PackedStorage) {
    fatalError("unsupported")
  }
  public static func - (_ lhs: PackedStorage, _ rhs: PackedStorage) -> PackedStorage {
    fatalError("unsupported")
  }
  public static func -= (_ lhs: inout PackedStorage, _ rhs: PackedStorage) {
    fatalError("unsupported")
  }
  public static var zero: PackedStorage {
    fatalError("unsupported")
  }
  public mutating func add(_ x: Double) {
    fatalError("unsupported")
  }
  public func adding(_ x: Double) -> PackedStorage {
    fatalError("unsupported")
  }
  public mutating func subtract(_ x: Double) {
    fatalError("unsupported")
  }
  public func subtracting(_ x: Double) -> PackedStorage {
    fatalError("unsupported")
  }
  public mutating func scale(by scalar: Double) {
    fatalError("unsupported")
  }
}

public struct Empty {}

public protocol VariableIDs {
  associatedtype Variable1
  associatedtype Variable2

  associatedtype MutableFastAssignments
  static func withMutableFastAssignments<R>(_ values: inout VariableAssignments, _ f: (inout MutableFastAssignments) -> R) -> R

  associatedtype FastAssignments
  static func withFastAssignments<R>(_ values: VariableAssignments, _ f: (FastAssignments) -> R) -> R
}

public struct MutableFastAssignments1<Variable1> {
  var buffer1: UnsafeMutableBufferPointer<Variable1>
  subscript(id: SimpleTypedID<Variable1>) -> Variable1 {
    _read {
      yield buffer1[id.perTypeID]
    }
    _modify {
      yield &buffer1[id.perTypeID]
    }
  }
}

public struct FastAssignments1<Variable1> {
  var buffer1: UnsafeBufferPointer<Variable1>
  subscript(id: SimpleTypedID<Variable1>) -> Variable1 {
    _read {
      yield buffer1[id.perTypeID]
    }
  }
}

public struct MutableFastAssignments2<Variable1, Variable2> {
  var buffer1: UnsafeMutableBufferPointer<Variable1>
  var buffer2: UnsafeMutableBufferPointer<Variable2>  // TODO: Is it a problem that these alias when Variable1 == Variable2?
  subscript(id: SimpleTypedID<Variable1>) -> Variable1 {
    _read {
      yield buffer1[id.perTypeID]
    }
    _modify {
      yield &buffer1[id.perTypeID]
    }
  }
  subscript(id: SimpleTypedID<Variable2>) -> Variable2 {
    _read {
      yield buffer2[id.perTypeID]
    }
    _modify {
      yield &buffer2[id.perTypeID]
    }
  }
}

public struct FastAssignments2<Variable1, Variable2> {
  var buffer1: UnsafeBufferPointer<Variable1>
  var buffer2: UnsafeBufferPointer<Variable2>  // TODO: Is it a problem that these alias when Variable1 == Variable2?
  subscript(id: SimpleTypedID<Variable1>) -> Variable1 {
    _read {
      yield buffer1[id.perTypeID]
    }
  }
  subscript(id: SimpleTypedID<Variable2>) -> Variable2 {
    _read {
      yield buffer2[id.perTypeID]
    }
  }
}

public struct VariableIDs1<Variable1>: VariableIDs {
  public typealias Variable2 = Empty
  public let id1: SimpleTypedID<Variable1>
  public typealias MutableFastAssignments = MutableFastAssignments1<Variable1>
  public static func withMutableFastAssignments<R>(_ values: inout VariableAssignments, _ f: (inout MutableFastAssignments) -> R) -> R {
    let buffer1 = values.homogeneousStorage[ObjectIdentifier(Variable1.self)]!.mutableBuffer.bindMemory(to: Variable1.self)
    var mutableFastAssignments = MutableFastAssignments1(buffer1: buffer1)
    return f(&mutableFastAssignments)
  }
  public typealias FastAssignments = FastAssignments1<Variable1>
  public static func withFastAssignments<R>(_ values: VariableAssignments, _ f: (FastAssignments) -> R) -> R {
    let buffer1 = values.homogeneousStorage[ObjectIdentifier(Variable1.self)]!.buffer.bindMemory(to: Variable1.self)
    return f(FastAssignments1(buffer1: buffer1))
  }
}

public struct VariableIDs2<Variable1, Variable2>: VariableIDs {
  public let id1: SimpleTypedID<Variable1>
  public let id2: SimpleTypedID<Variable2>
  public typealias MutableFastAssignments = MutableFastAssignments2<Variable1, Variable2>
  public static func withMutableFastAssignments<R>(_ values: inout VariableAssignments, _ f: (inout MutableFastAssignments) -> R) -> R {
    let buffer1 = values.homogeneousStorage[ObjectIdentifier(Variable1.self)]!.mutableBuffer.bindMemory(to: Variable1.self)
    let buffer2 = values.homogeneousStorage[ObjectIdentifier(Variable2.self)]!.mutableBuffer.bindMemory(to: Variable2.self)
    var mutableFastAssignments = MutableFastAssignments2(buffer1: buffer1, buffer2: buffer2)
    return f(&mutableFastAssignments)
  }
  public typealias FastAssignments = FastAssignments2<Variable1, Variable2>
  public static func withFastAssignments<R>(_ values: VariableAssignments, _ f: (FastAssignments) -> R) -> R {
    let buffer1 = values.homogeneousStorage[ObjectIdentifier(Variable1.self)]!.buffer.bindMemory(to: Variable1.self)
    let buffer2 = values.homogeneousStorage[ObjectIdentifier(Variable2.self)]!.buffer.bindMemory(to: Variable2.self)
    return f(FastAssignments2(buffer1: buffer1, buffer2: buffer2))
  }
}

// public struct VariableIDs2a<Variable1>: VariableIDs {
//   public typealias Variable2 = Variable1
//   public let id1: SimpleTypedID<Variable1>
//   public let id2: SimpleTypedID<Variable2>
//   public typealias T = MutableFastAssignments1<Variable1>
//   public static func withFastAssignments<R>(_ values: VariableAssignments, _ f: (T) -> R) -> R {
//     let buffer1 = values.homogeneousStorage[ObjectIdentifier(Variable1.self)]!.mutableBuffer.bindMemory(to: Variable1.self)
//     return f(MutableFastAssignments1(buffer1: buffer1))
//   }
// }

// struct VariableIDs2<Variable1, Variable2>: VariableIDs {
//   let id1: SimpleTypedID<Variable1>
//   let id2: SimpleTypedID<Variable2>
//   typealias T = FastAssignments2<Variable1, Variable2>
//   static func withFastAssignments<R>(_ values: inout VariableAssignments, _ f: (inout T) -> R) -> R{
//     let buffer1 = values.homogeneousStorage[ObjectIdentifier(Variable1.self)].bindMemory(to: Variable1.self).mutableBuffer
//     let buffer2 = values.homogeneousStorage[ObjectIdentifier(Variable1.self)].bindMemory(to: Variable2.self).mutableBuffer
//     return f(FastAssignments2(buffer1: buffer1, buffer2: buffer2))
//   }
// }

public protocol AnyFactor {
  associatedtype InputIDs: VariableIDs
  var inputIDs: InputIDs { get }

  func error(at value: InputIDs.FastAssignments) -> Double

  associatedtype ErrorVector: EuclideanVectorSpace
  func errorVector(at value: InputIDs.FastAssignments) -> ErrorVector

  associatedtype LinearizedFactor: AnyFactor
  func linearized(at value: InputIDs.FastAssignments) -> LinearizedFactor

  /// Applies this Gaussian factor's forward linear map to `x`.
  func applyLinearForward(_ x: InputIDs.FastAssignments) -> ErrorVector

  /// Applies this Gaussian factor's transpose linear map to `y`.
  func applyLinearTranspose(_ y: ErrorVector, into result: inout InputIDs.MutableFastAssignments)
}

public protocol Factor1: AnyFactor where InputIDs == VariableIDs1<Variable1> {
  associatedtype Variable1
  var inputId1: SimpleTypedID<Variable1> { get }

  func error(at value: Variable1) -> Double

  func errorVector(at value: Variable1) -> ErrorVector

  func linearized(at value: Variable1) -> LinearizedFactor

  /// Applies this Gaussian factor's forward linear map to `x`.
  func applyLinearForward(_ x: Variable1) -> ErrorVector

  /// Applies this Gaussian factor's transpose linear map to `y`.
  func applyLinearTranspose(_ y: ErrorVector, into result: inout Variable1)
}

extension Factor1 {
  var inputIDs: VariableIDs1<Variable1> {
    return VariableIDs1(id1: inputId1)
  }

  func error(at value: FastAssignments1<Variable1>) -> Double {
    return error(at: value[inputId1])
  }

  func errorVector(at value: FastAssignments1<Variable1>) -> ErrorVector {
    return errorVector(at: value[inputId1])
  }

  func linearized(at value: FastAssignments1<Variable1>) -> LinearizedFactor {
    return linearized(at: value[inputId1])
  }

  func applyLinearForward(_ x: FastAssignments1<Variable1>) -> ErrorVector {
    return applyLinearForward(x[inputId1])
  }

  func applyLinearTranspose(_ y: ErrorVector, into result: inout MutableFastAssignments1<Variable1>) {
    applyLinearTranspose(y, into: &result[inputId1])
  }
}

extension Factor1 {
  func error(at value: Variable1) -> Double {
    return errorVector(at: value).squaredNorm
  }

  /// Applies this Gaussian factor's forward linear map to `x`.
  func applyLinearForward(_ x: Variable1) -> ErrorVector {
    fatalError("not supported")
  }

  /// Applies this Gaussian factor's transpose linear map to `y`.
  func applyLinearTranspose(_ y: ErrorVector, into result: inout Variable1) {
    fatalError("not supported")
  }
}

public protocol Factor2: AnyFactor where InputIDs == VariableIDs2<Variable1, Variable2> {
  associatedtype Variable1
  associatedtype Variable2
  var inputId1: SimpleTypedID<Variable1> { get }
  var inputId2: SimpleTypedID<Variable2> { get }

  func error(at value1: Variable1, _ value2: Variable2) -> Double

  func errorVector(at value1: Variable1, _ value2: Variable2) -> ErrorVector

  func linearized(at value1: Variable1, _ value2: Variable2) -> LinearizedFactor

  /// Applies this Gaussian factor's forward linear map to `x`.
  func applyLinearForward(_ x1: Variable1, _ x2: Variable2) -> ErrorVector

  /// Applies this Gaussian factor's transpose linear map to `y`.
  func applyLinearTranspose(_ y: ErrorVector, into result1: inout Variable1, _ result2: inout Variable2)
}

extension Factor2 {
  var inputIDs: VariableIDs2<Variable1, Variable2> {
    return VariableIDs2(id1: inputId1, id2: inputId2)
  }

  func error(at value: FastAssignments2<Variable1, Variable2>) -> Double {
    return error(at: value[inputId1], value[inputId2])
  }

  func errorVector(at value: FastAssignments2<Variable1, Variable2>) -> ErrorVector {
    return errorVector(at: value[inputId1], value[inputId2])
  }

  func linearized(at value: FastAssignments2<Variable1, Variable2>) -> LinearizedFactor {
    return linearized(at: value[inputId1], value[inputId2])
  }

  func applyLinearForward(_ x: FastAssignments2<Variable1, Variable2>) -> ErrorVector {
    return applyLinearForward(x[inputId1], x[inputId2])
  }

  func applyLinearTranspose(_ y: ErrorVector, into result: inout MutableFastAssignments2<Variable1, Variable2>) {
    applyLinearTranspose(y, into: &result.buffer1[inputId1.perTypeID], &result.buffer2[inputId2.perTypeID])
  }
}

extension Factor2 {
  func error(at value1: Variable1, _ value2: Variable2) -> Double {
    return errorVector(at: value1, value2).squaredNorm
  }

  /// Applies this Gaussian factor's forward linear map to `x`.
  func applyLinearForward(_ x1: Variable1, _ x2: Variable2) -> ErrorVector {
    fatalError("not supported")
  }

  /// Applies this Gaussian factor's transpose linear map to `y`.
  func applyLinearTranspose(_ y: ErrorVector, into result1: inout Variable1, _ result2: inout Variable2) {
    fatalError("not supported")
  }
}