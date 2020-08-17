import Foundation
import PenguinTesting
import TensorFlow
import XCTest

import SwiftFusion

extension Vector {
  /// XCTests `self`'s semantic conformance to `Vector`, expecting its scalars to match
  /// `expectedScalars`.
  ///
  /// Note: This does check the semantics of the vector math operations, but only using vectors
  /// that have the same `scalar.indices` as `self`. If `Self` supports vector operations between
  /// instances between different `scalars.indices`, then tests should additionally call
  /// `checkVectorMathSemantics` with examples of vectors with differing `scalars.indices`.
  ///
  /// - Parameter doesNotSupportTwoScalars: pass `true` when `Self` does not have instances with
  ///   `dimension >= 2`.
  /// - Requires: `!distinctScalars.elementsEqual(self.scalars)`.
  /// - Complexity: O(N²), where N is `self.dimension`.
  func checkVectorSemantics<S1: Collection, S2: Collection>(
    expecting expectedScalars: S1,
    writing distinctScalars: S2,
    doesNotSupportTwoScalars: Bool = false
  ) where S1.Element == Double, S2.Element == Double {
    self.scalars.checkCollectionSemantics(
      expecting: expectedScalars, doesNotSupportTwoElements: doesNotSupportTwoScalars)

    var mutableScalars = self.scalars
    mutableScalars.checkMutableCollectionSemantics(writing: distinctScalars)

    XCTAssertEqual(self.dimension, expectedScalars.count)

    // Check that setting `scalars` actually changes it.
    var mutableSelf = self
    for (i, e) in zip(mutableSelf.scalars.indices, distinctScalars) { mutableSelf.scalars[i] = e }
    XCTAssertTrue(mutableSelf.scalars.elementsEqual(distinctScalars))

    // Returns a vector with the same `scalars.indices` as `self` but with the scalars replaced
    // by the stride from `start` by `stride`.
    func stride(from start: Double, by stride: Double) -> Self {
      var r = self
      for (i, e) in zip(
        r.scalars.indices,
        Swift.stride(from: start, to: start + Double(r.dimension) * stride, by: stride)
      ) {
        r.scalars[i] = e
      }
      return r
    }

    stride(from: 1, by: 1).checkVectorMathSemantics(
      other: stride(from: 10, by: 10),
      expectedPlusOther: stride(from: 11, by: 11),
      expectedMinusOther: stride(from: -9, by: -9),
      expectedDotOther:
        Double(10 * self.dimension * (self.dimension + 1) * (2 * self.dimension + 1) / 6),
      scaleFactor: 7,
      expectedScaled: stride(from: 7, by: 7))

    self.withUnsafeBufferPointer { b in
      XCTAssertEqual(b.count, expectedScalars.count)
      for (actual, expected) in zip(b, expectedScalars) {
        XCTAssertEqual(actual, expected)
      }
    }

    mutableSelf = self
    mutableSelf.withUnsafeMutableBufferPointer { b in
      for (i, j) in zip(b.indices, distinctScalars.indices) {
        b[i] = distinctScalars[j]
      }
    }
    XCTAssertTrue(mutableSelf.scalars.elementsEqual(distinctScalars))
  }

  /// XCTests the semantics of vector math operations on `self`.
  public func checkVectorMathSemantics(
    other: Self,
    expectedPlusOther: Self,
    expectedMinusOther: Self,
    expectedDotOther: Double,
    scaleFactor: Double,
    expectedScaled: Self
  ) {
    let zero = self.zeroTangentVector

    zero.checkPlus(zero, equals: zero)
    self.checkPlus(zero, equals: self)
    zero.checkPlus(self, equals: self)
    self.checkPlus(-self, equals: zero)

    zero.checkMinus(zero, equals: zero)
    self.checkMinus(zero, equals: self)
    zero.checkMinus(self, equals: -self)
    self.checkMinus(self, equals: zero)

    self.checkTimes(0, equals: zero)
    self.checkTimes(1, equals: self)
    self.checkTimes(-1, equals: -self)

    self.checkPlus(other, equals: expectedPlusOther)
    other.checkPlus(self, equals: expectedPlusOther)

    self.checkMinus(other, equals: expectedMinusOther)
    other.checkMinus(self, equals: -expectedMinusOther)

    self.checkTimes(scaleFactor, equals: expectedScaled)

    self.checkDot(other, equals: expectedDotOther)
    other.checkDot(self, equals: expectedDotOther)
  }

  /// XCTests the semantics of the mutating and nonmutating addition operations at `(self, other)`,
  /// and their derivatives.
  private func checkPlus(_ other: Self, equals expectedResult: Self) {
    // Do the checks in a closure so that we can check both the mutating and nonmutating versions.
    func check(_ f: @differentiable (Self, Self) -> Self) {
      XCTAssertEqual(f(self, other), expectedResult)

      let (result, pb) = valueWithPullback(at: self, other, in: f)
      XCTAssertEqual(result, expectedResult)
      for v in result.unitVectors {
        XCTAssertEqual(pb(v).0, v)
        XCTAssertEqual(pb(v).1, v)
      }
    }

    check { $0 + $1 }
    check {
      var r = $0
      r += $1
      return r
    }
  }

  /// XCTests the semantics of the mutating and nonmutating subtraction operations at `(self, other)`,
  /// and their derivatives.
  private func checkMinus(_ other: Self, equals expectedResult: Self) {
    // Do the checks in a closure so that we can check both the mutating and nonmutating versions.
    func check(_ f: @differentiable (Self, Self) -> Self) {
      XCTAssertEqual(f(self, other), expectedResult)

      let (result, pb) = valueWithPullback(at: self, other, in: f)
      XCTAssertEqual(result, expectedResult)
      for v in result.unitVectors {
        XCTAssertEqual(pb(v).0, v)
        XCTAssertEqual(pb(v).1, -v)
      }
    }

    check { $0 - $1 }
    check {
      var r = $0
      r -= $1
      return r
    }
  }

  /// XCTests the semantics of the mutating and nonmutating scalar multiplication operations at
  /// `(self, scaleFactor)`, and their derivatives.
  private func checkTimes(_ scaleFactor: Double, equals expectedResult: Self) {
    // Do the checks in a closure so that we can check both the mutating and nonmutating versions.
    func check(_ f: @differentiable (Double, Self) -> Self) {
      XCTAssertEqual(f(scaleFactor, self), expectedResult)

      let (result, pb) = valueWithPullback(at: scaleFactor, self, in: f)
      XCTAssertEqual(result, expectedResult)
      for v in result.unitVectors {
        XCTAssertEqual(pb(v).0, v.dot(self))
        XCTAssertEqual(pb(v).1, scaleFactor * v)
      }
    }

    check { $0 * $1 }
    check {
      var r = $1
      r *= $0
      return r
    }
  }

  /// XCTests the semantics of the inner product operation at `(self, other)`, and its derivative.
  private func checkDot(_ other: Self, equals expectedResult: Double) {
    XCTAssertEqual(self.dot(other), expectedResult)

    let (result, pb) = valueWithPullback(at: self, other) { $0.dot($1) }
    XCTAssertEqual(result, expectedResult)
    XCTAssertEqual(pb(1).0, other)
    XCTAssertEqual(pb(1).1, self)
  }

  /// For each `i` in `self.scalars.indices`, a vectors with the same `scalars.indices` as `self` but
  /// with a `1` at `i` and `0`s at all other indices.
  private var unitVectors: LazyMapCollection<Scalars.Indices, Self> {
    self.scalars.indices.lazy.map { i in
      var r = self.zeroTangentVector
      r.scalars[i] = 1
      return r
    }
  }
}

class VectorConversionTests: XCTestCase {
  /// Tests converting from one type to another type with the same number of elements.
  func testConversion() {
    let v = Vector9(0, 1, 2, 3, 4, 5, 6, 7, 8)
    let m = Matrix3(0, 1, 2, 3, 4, 5, 6, 7, 8)
    XCTAssertEqual(Vector9(m), v)

    let (value, pb) = valueWithPullback(at: m) { Vector9($0) }
    XCTAssertEqual(value, v)
    for (bV, bM) in zip(Vector9.standardBasis, Matrix3.standardBasis) {
      XCTAssertEqual(pb(bV), bM)
    }
  }

  /// Tests concatenating two vectors.
  func testConcatenate() {
    let v1 = Vector2(0, 1)
    let v2 = Vector3(2, 3, 4)
    let expected = Vector5(0, 1, 2, 3, 4)
    XCTAssertEqual(Vector5(concatenating: v1, v2), expected)

    let (value, pb) = valueWithPullback(at: v1, v2) { Vector5(concatenating: $0, $1) }
    XCTAssertEqual(value, expected)

    XCTAssertEqual(pb(Vector5(1, 0, 0, 0, 0)).0, Vector2(1, 0))
    XCTAssertEqual(pb(Vector5(0, 1, 0, 0, 0)).0, Vector2(0, 1))
    XCTAssertEqual(pb(Vector5(0, 0, 1, 0, 0)).0, Vector2(0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 1, 0)).0, Vector2(0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 0, 1)).0, Vector2(0, 0))

    XCTAssertEqual(pb(Vector5(1, 0, 0, 0, 0)).1, Vector3(0, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 1, 0, 0, 0)).1, Vector3(0, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 1, 0, 0)).1, Vector3(1, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 1, 0)).1, Vector3(0, 1, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 0, 1)).1, Vector3(0, 0, 1))
  }

  func testConvertToTensor() {
    let v = Vector3(1, 2, 3)
    let expectedT = Tensor<Double>([1, 2, 3])
    XCTAssertEqual(v.flatTensor, expectedT)

    let (value, pb) = valueWithPullback(at: v) { $0.flatTensor }
    XCTAssertEqual(value, expectedT)
    XCTAssertEqual(pb(Tensor([1, 0, 0])), Vector3(1, 0, 0))
    XCTAssertEqual(pb(Tensor([0, 1, 0])), Vector3(0, 1, 0))
    XCTAssertEqual(pb(Tensor([0, 0, 1])), Vector3(0, 0, 1))
  }

  func testConvertFromTensor() {
    let t = Tensor<Double>([1, 2, 3])
    let expectedV = Vector3(1, 2, 3)
    XCTAssertEqual(Vector3(flatTensor: t), expectedV)

    let (value, pb) = valueWithPullback(at: t) { Vector3(flatTensor: $0) }
    XCTAssertEqual(value, expectedV)
    XCTAssertEqual(pb(Vector3(1, 0, 0)), Tensor([1, 0, 0]))
    XCTAssertEqual(pb(Vector3(0, 1, 0)), Tensor([0, 1, 0]))
    XCTAssertEqual(pb(Vector3(0, 0, 1)), Tensor([0, 0, 1]))
  }
}
