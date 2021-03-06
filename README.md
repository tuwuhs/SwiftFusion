# SwiftFusion

Differentiable Swift based sensor fusion library. 

Think factor graphs a la [GTSAM](https://gtsam.org/) coupled with deep learning, [Jax](https://github.com/google/jax)/[TensorFlow](https://www.tensorflow.org/) style. Based on [Swift for TensorFlow](https://www.tensorflow.org/swift).

Still very early, but feel free to explore! Subject to *massive* change :-)

## Getting Started

### Swift on MacOS

Using XCode is the easiest way to develop on Mac. Unfortunately, the current version of Xcode 12 has a bug, and you need to work with the XCode beta, available from Apple [here](https://developer.apple.com/download/).

Follow the instructions to [install Swift for TensorFlow on MacOS](https://github.com/tensorflow/swift/blob/master/Installation.md#macos). Installing the latest development snapshot is recommended.

To use experimental toolchain:
```
export PATH="/Library/Developer/Toolchains/swift-tensorflow-RELEASE-0.11.xctoolchain/usr/bin/:$PATH"
```

To re-generate XCode project:
```
swift package generate-xcodeproj
```

### Installing Swift on Linux

Requirements: Ubuntu 18.04 (if you use GPU). 

Follow the instructions to [install Swift for TensorFlow on Linux](https://github.com/tensorflow/swift/blob/master/Installation.md#linux).  Installing the latest development snapshot is recommended.

### Installing S4TF on Jetson Devices

Download toolchain [here](https://storage.googleapis.com/swift-tensorflow-artifacts/oneoff-builds/swift-tensorflow-RELEASE-0.11-Jetson4.4.tar.gz), and then follow Linux instructions.

### Run tests

To check whether everything works you can run all the tests by changing to the SwiftFusion directory and
```
swift test --enable-test-discovery
```

### Run benchmarks

```
swift run -c release -Xswiftc -cross-module-optimization SwiftFusionBenchmarks
```

## Working with VS Code

To enable autocomplete in VSCode, install the plugin vknabel.vscode-swift-development-environment, and set the following plugin settings:

- "sde.languageServerMode": "sourcekit-lsp",
- "sourcekit-lsp.serverPath": "<your toolchain path>/usr/bin/sourcekit-lsp",
- "sourcekit-lsp.toolchainPath": "<your toolchain path>",
- "swift.path.swift_driver_bin": "<your toolchain path>/usr/bin/swift",

Debugging within VS code is easiest via the CodeLLDB plugin so you can debug in vscode. You need to set the following setting:
"lldb.library": "/swift-tensorflow-toolchain/usr/lib/liblldb.so"

A sample launch.json file:

```json
{
"version": "0.2.0",
"configurations": [
{
"type": "lldb",
"request": "launch",
"name": "Debug",
"program": "${workspaceFolder}/.build/x86_64-unknown-linux-gnu/debug/SwiftFusionPackageTests.xctest",
"args": ["--enable-test-discovery"],
"cwd": "${workspaceFolder}"
}
]
}
```

## Code Overview

The main code is in Sources/SwiftFusion, which as a number of sub-directories:

### Core
The main protocols on which SwiftFusion is built:
- Vector.swift: protocol that formalizes a Euclidean vector space with standard orthonormal basis, and defines a great number of default methods for it.  Also defines the `ScalarsInitializableVector` and `FixedSizeVector` protocols, and a collection `StandardBasis<V: Vector>` for the orthogonal bases.
- Manifold.swift: protocol that inherits from `Differentiable` that adds `retract` and `localCoordinate`, which convert between a `Manifold` structure and its `LocalCoordinate`
- LieGroup.swift: protocol that formalizes making a differentiable manifold into a Lie group with composition operator `*`. Also defines `LieGroupCoordinate` protocol.

Some specific Vector-like data structures:
- VectorN.swift: generated by VectorN.swift.gyb, implements small fixed vector structs conforming to `AdditiveArithmetic`, `Vector`, and `FixedSizeVector`.
- FixedSizeMatrix.swift: matrix whose dimensions are known at compile time. Conforms to `Equatable, KeyPathIterable, CustomStringConvertible, AdditiveArithmetic, Differentiable, FixedSizeVector`

A number of utilities:
- DataTypes.swift: for now, just implements `LieGroup` protocol for `Vector5`
- Dictionary+Differentiable.swift: makes `Dictionary` differentiable
- MathUtil.swift: for now just `pinv` (pseudo-inverse), using [Tensor.svd](https://www.tensorflow.org/swift/api_docs/Structs/Tensor#svdcomputeuv:fullmatrices:)
- TensorVector.swift: a view of a `Tensor` that conforms to the `Vector` protocol.
- TrappingDouble.swift: a wrapper for `Double` that traps instead of allowing `NaN`.
- Tuple+Vector.swift: (undocumented)
- TypeKeyedArrayBuffers.swift: related to storage of Values/Factors

### Inference
- FactorGraph.swift: the main `FactorGraph` struct, which stores factors
- Factor.swift: defines the core factor protocol hierarchy `GaussianFactor` : `LinearizableFactor` : `VectorFactor` : `Factor`, as well as the `VariableTuple` and `DifferentiableVariableTuple` protocols, as well as extensions to make `Tuple` conform to them.
- GaussianFactorGraph.swift: A factor graph whose factors are all `GaussianFactor`s.
A factor graph stores factors as a storage array of type `[ObjectIdentifier: AnyFactorArrayBuffer]`, where [ObjectIdentifier](https://developer.apple.com/documentation/swift/objectidentifier) is a builtt-in Swift identifier type, and `AnyFactorArrayBuffer` is defined in [FactorsStorage.swift](https://github.com/borglab/SwiftFusion/blob/master/Sources/SwiftFusion/Inference/FactorsStorage.swift). It's complicated.

### Optimizers
- GradientDescent.swift: a basic gradient descent optimizer
- CGLS.swift: Conjugate Gradient solver for Least-squares problems
- LM.swift: a Levenberg-Marquardt nonlinear optimizer

### MCMC
- RandomWalkMetropolis.swift: implements a Metropolis sampler modeled after [its equivalent](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/RandomWalkMetropolis) in [TensorFlow Probability](https://www.tensorflow.org/probability)

### Geometry
For 2D and 3D points we just use Vector2 and Vector3, but here we define 2D and 3D rotations, and 2D and 3D poses. We use the GTSAM naming scheme: `Rot`, `Pose2`, `Rot3`, and `Pose3`.

### Image
- ArrayImage.swift: `Differentiable` multi-channel image stored as `[Double]`. Has a differentiable `tensor` method and `update` function.
- OrientedBoundingBox.swift: a rectangular region of an image, not necessarily axis-aligned.
- Patch.swift: a differentiable `patch` method for `ArrayImage`, that returns a resampled image for the given `OrientedBoundingBox`.

### Datasets
Mostly reading of G2O files

# LICENSE

Copyright 2020 The SwiftFusion Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
