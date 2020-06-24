import Foundation
import TensorFlow
import XCTest

import PenguinStructures
@testable import SwiftFusion

/// A factor on two discrete labels evaluation the transition probability
struct DiscreteTransitionFactor : Factor {
  typealias Variables = Tuple2<Int, Int>
  
  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices
  
  /// The number of states.
  let stateCount: Int
  
  /// Entry `i * stateCount + j` is the probability of transitioning from state `j` to state `i`.
  let transitionMatrix: [Double]
  
  init(
    _ inputId1: TypedID<Int, Int>,
    _ inputId2: TypedID<Int, Int>,
    _ stateCount: Int,
    _ transitionMatrix: [Double]
  ) {
    precondition(transitionMatrix.count == stateCount * stateCount)
    self.edges = Tuple2(inputId1, inputId2)
    self.stateCount = stateCount
    self.transitionMatrix = transitionMatrix
  }
  
  func error(at q: Variables) -> Double {
    let (label1, label2) = (q.head, q.tail.head)
    return -log(transitionMatrix[label2 * stateCount + label1])
  }
}

/// A factor with a switchable motion model.
///
/// `JacobianRows` specifies the `Rows` parameter of the Jacobian of this factor. See the
/// documentation on `JacobianFactor.jacobian` for more information. Use the typealiases below to
/// avoid specifying this type parameter every time you create an instance.
public struct SwitchingBetweenFactor<Pose: LieGroup, JacobianRows: FixedSizeArray>:
  LinearizableFactor
where JacobianRows.Element == Tuple2<Pose.TangentVector, Pose.TangentVector>
{
  public typealias Variables = Tuple3<Pose, Int, Pose>
  
  public let edges: Variables.Indices
  
  /// Movement temmplates for each label.
  let movements: [Pose]
  
  public init(_ from: TypedID<Pose, Int>,
              _ label: TypedID<Int, Int>,
              _ to: TypedID<Pose, Int>,
              _ movements: [Pose]) {
    self.edges = Tuple3(from, label, to)
    self.movements = movements
  }
  
  public typealias ErrorVector = Pose.TangentVector
  
  @differentiable(wrt: (start, end))
  public func errorVector(_ start: Pose, _ label: Int, _ end: Pose) -> ErrorVector {
    let actualMotion = between(start, end)
    return movements[label].localCoordinate(actualMotion)
  }
  
  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }
  
  public func errorVector(at x: Variables) -> Pose.TangentVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head)
  }
  
  public typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    let (start, label, end) = (x.head, x.tail.head, x.tail.tail.head)
    let (startEdge, endEdge) = (edges.head, edges.tail.tail.head)
    let differentiableVariables = Tuple2(start, end)
    let differentiableEdges = Tuple2(startEdge, endEdge)
    return Linearization(
      linearizing: { differentiableVariables in
        let (start, end) = (differentiableVariables.head, differentiableVariables.tail.head)
        return errorVector(start, label, end)
      },
      at: differentiableVariables,
      edges: differentiableEdges
    )
  }
}

/// A between factor on `Pose2`.
public typealias SwitchingBetweenFactor2 = SwitchingBetweenFactor<Pose2, Array3<Tuple2<Vector3, Vector3>>>

/// A between factor on `Pose3`.
public typealias SwitchingBetweenFactor3 = SwitchingBetweenFactor<Pose3, Array6<Tuple2<Vector6, Vector6>>>

class Scratch: XCTestCase {
  let origin = Pose2(0,0,0)
  let forwardMove = Pose2(1,0,0)
  let (z1, z2, z3) = (Pose2(10, 0, 0),Pose2(11, 0, 0),Pose2(12, 0, 0))
  var expectedTrackingError : Double = 0.0
  override func setUp() {
    super.setUp()
    expectedTrackingError = 2 * origin.localCoordinate(z1).squaredNorm
      +  origin.localCoordinate(z2).squaredNorm
      +  origin.localCoordinate(z3).squaredNorm
      +  2 * origin.localCoordinate(forwardMove).squaredNorm
  }
  func createTrackingFactorGraph() -> (FactorGraph, VariableAssignments) {
    var variables = VariableAssignments()
    let x1 = variables.store(origin)
    let x2 = variables.store(origin)
    let x3 = variables.store(origin)
    
    var graph = FactorGraph()
    graph.store(PriorFactor2(x1, z1)) // prior
    graph.store(PriorFactor2(x1, z1))
    graph.store(PriorFactor2(x2, z2))
    graph.store(PriorFactor2(x3, z3))
    graph.store(BetweenFactor2(x1, x2, forwardMove))
    graph.store(BetweenFactor2(x2, x3, forwardMove))
    
    return (graph, variables)
  }
  
  func printPoses(_ variables : VariableAssignments) {
    print(variables[TypedID<Pose2, Int>(0)])
    print(variables[TypedID<Pose2, Int>(1)])
    print(variables[TypedID<Pose2, Int>(2)])
  }
  
  /// Tracking example from Figure 2.a
  func testTrackingExample() {
    // create a factor graph
    var (graph, variables) = createTrackingFactorGraph()
    _ = graph as FactorGraph
    
    // check number of factor types
    XCTAssertEqual(graph.storage.count, 2)
    
    // check error at initial estimate
    XCTAssertEqual(graph.error(at: variables), expectedTrackingError)
    
    // optimize
    var opt = LM()
    try! opt.optimize(graph: graph, initial: &variables)
    
    // print
    printPoses(variables)
  }
  
  func createSwitchingFactorGraph() -> (FactorGraph, VariableAssignments) {
    var variables = VariableAssignments()
    let x1 = variables.store(origin)
    let x2 = variables.store(origin)
    let x3 = variables.store(origin)
    let q1 = variables.store(0)
    let q2 = variables.store(0)
    
    // Model parameters.
    let labelCount = 3
    let transitionMatrix: [Double] = [
      0.8, 0.1, 0.1,
      0.1, 0.8, 0.1,
      0.1, 0.1, 0.8
    ]
    let movements = [
      forwardMove,          // go forwards
      Pose2(1, 0, .pi / 4), // turn left
      Pose2(1, 0, -.pi / 4) // turn right
    ]
    
    var graph = FactorGraph()
    graph.store(PriorFactor2(x1, z1)) // prior
    graph.store(PriorFactor2(x1, z1))
    graph.store(PriorFactor2(x2, z2))
    graph.store(PriorFactor2(x3, z3))
    graph.store(SwitchingBetweenFactor2(x1, q1, x2, movements))
    graph.store(SwitchingBetweenFactor2(x2, q2, x3, movements))
    graph.store(DiscreteTransitionFactor(q1, q2, labelCount, transitionMatrix))
    
    return (graph, variables)
  }
  
  func printLabels(_ variables : VariableAssignments) {
    print(variables[TypedID<Int, Int>(0)])
    print(variables[TypedID<Int, Int>(1)])
  }
  
  /// Tracking switching from Figure 2.b
  func testSwitchingExample() {
    // create a factor graph
    var (graph, variables) = createSwitchingFactorGraph()
    _ = graph as FactorGraph
    _ = variables as VariableAssignments
    
    // check number of factor types
    XCTAssertEqual(graph.storage.count, 3)
    
    // check error at initial estimate, allow slack to account for discrete transition
    XCTAssertEqual(graph.error(at: variables), 467.0, accuracy:0.3)
    
    // optimize
    var opt = LM()
    try! opt.optimize(graph: graph, initial: &variables)
    
    
    // print
    printLabels(variables)
    printPoses(variables)

    // Create initial state for MCMC sampler
    let current_state = variables
    
    // Do MCMC the tfp way
    let num_results = 50
    let num_burnin_steps = 30
    
    /// Proposal to change one label, and re-optimize
    let flipAndOptimize = {(x:VariableAssignments) -> VariableAssignments in
      let labelVars = x.storage[ObjectIdentifier(Int.self)]
      let positionVars = x.storage[ObjectIdentifier(Pose2.self)]
      
      // Randomly change one label.
      let i = Int.random(in: 0..<labelVars!.count)
      let id = TypedID<Int, Int>(i)
      var y = x
      y[id] = Int.random(in: 0..<3)
      
      // Initialize trajectory to zero before optimizing
      for i in 0..<positionVars!.count {
        y[TypedID<Pose2,Int>(i)] = Pose2(0, 0, 0)
      }
      
      // Pose2SLAM to find new proposed positions.
      self.printLabels(y)
      self.printPoses(y)
      try! opt.optimize(graph: graph, initial: &y)
      self.printLabels(y)
      self.printPoses(y)
      return y
    }
    
    let kernel = RandomWalkMetropolis(
      target_log_prob_fn: {(x:VariableAssignments) in 0.0},
      new_state_fn: flipAndOptimize
    )
    
    let states = sampleChain(
      num_results,
      current_state,
      kernel,
      num_burnin_steps
    )
    _ = states as Array
    XCTAssertEqual(states.count, num_results)
    print(states)
  }
}