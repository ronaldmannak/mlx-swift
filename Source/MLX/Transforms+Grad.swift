import Foundation
import Cmlx

// This file is generated by GenerateGrad.

// Returns a function which computes the gradient of `f`.
public func grad(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: [Int] = [0]) -> ([MLXArray]) -> [MLXArray] {
    // Converts the given function `f()` into canonical types, e.g.
    // (MLXArray) -> MLXArray into the canonical form ([MLXArray]) -> [MLXArray].
    //
    // First use the wrapArguments() and wrapResult() function to transform
    // it into that form.  Then call buildValueAndGradient() to produce a new
    // function with the same canonical form.
    //
    // Finally use unwrapArguments() and unwrapResult() to transform the function
    // back into the original signature.
    //
    // Note: this particular form of the function is already in the canonical
    // form and the wrap/unwrap calls are identity functions.
    let wrappedFunction = wrapResult(wrapArguments(f))
    let gradientFunction = buildGradient(wrappedFunction, argumentNumbers: argumentNumbers)
    let uag: ([MLXArray]) -> [MLXArray] = unwrapArguments(gradientFunction)
    return unwrapResult(uag)
}

// See ``grad(_:)-r8dv``
public func grad(_ f: @escaping ([MLXArray]) -> MLXArray, argumentNumbers: [Int] = [0]) -> ([MLXArray]) -> MLXArray {
    let wrappedFunction = wrapResult(wrapArguments(f))
    let gradientFunction = buildGradient(wrappedFunction, argumentNumbers: argumentNumbers)
    let uag: ([MLXArray]) -> [MLXArray] = unwrapArguments(gradientFunction)
    return unwrapResult(uag)
}

// See ``grad(_:)-r8dv``
public func grad(_ f: @escaping (MLXArray) -> [MLXArray]) -> (MLXArray) -> [MLXArray] {
    let wrappedFunction = wrapResult(wrapArguments(f))
    let gradientFunction = buildGradient(wrappedFunction, argumentNumbers: [0])
    let uag: (MLXArray) -> [MLXArray] = unwrapArguments(gradientFunction)
    return unwrapResult(uag)
}

// See ``grad(_:)-r8dv``
public func grad(_ f: @escaping (MLXArray) -> MLXArray) -> (MLXArray) -> MLXArray {
    let wrappedFunction = wrapResult(wrapArguments(f))
    let gradientFunction = buildGradient(wrappedFunction, argumentNumbers: [0])
    let uag: (MLXArray) -> [MLXArray] = unwrapArguments(gradientFunction)
    return unwrapResult(uag)
}

// Returns a function which computes the value and gradient of `f`.
public func valueAndGrad(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: [Int] = [0]) -> ([MLXArray]) -> ([MLXArray], [MLXArray]) {
    //
    return buildValueAndGradient(f, argumentNumbers: argumentNumbers)
}

// Returns a function which computes the value and gradient of `f`.
public func valueAndGradNested(_ f: @escaping (NestedDictionary<String, MLXArray>, [MLXArray]) -> [MLXArray], argumentNumbers: [Int] = [0]) -> (NestedDictionary<String, MLXArray>, [MLXArray]) -> ([MLXArray], [MLXArray]) {
    //
    return buildValueAndGradient(f, argumentNumbers: argumentNumbers)
}


// MARK: - Functions to wrap and unwrap types in closures

@inline(__always)
private func wrapArguments<Result>(_ f: @escaping ([MLXArray]) -> Result) -> ([MLXArray]) -> Result {
    f
}

@inline(__always)
private func wrapResult(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
    f
}

@inline(__always)
private func wrapResult(_ f: @escaping ([MLXArray]) -> MLXArray) -> ([MLXArray]) -> [MLXArray] {
    { (arrays: [MLXArray]) in
        [f(arrays)]
    }
}

@inline(__always)
private func unwrapArguments(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
    f
}

@inline(__always)
private func unwrapResult(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
    f
}

@inline(__always)
private func wrapArguments<Result>(_ f: @escaping (MLXArray) -> Result) -> ([MLXArray]) -> Result {
    { (arrays: [MLXArray]) in
        f(arrays[0])
    }
}

@inline(__always)
private func unwrapArguments(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> (MLXArray) -> [MLXArray] {
    { (a0: MLXArray) in
        f([a0])
    }
}

@inline(__always)
private func unwrapResult(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> (MLXArray) {
    { (a0: [MLXArray]) in
        let v = f(a0)
        return (v[0])
    }
}

@inline(__always)
private func unwrapResult(_ f: @escaping (MLXArray) -> [MLXArray]) -> (MLXArray) -> [MLXArray] {
    { (a0: MLXArray) in
        f(a0)
    }
}

@inline(__always)
private func unwrapResult(_ f: @escaping (MLXArray) -> [MLXArray]) -> (MLXArray) -> (MLXArray) {
    { (a0: MLXArray) in
        let v = f(a0)
        return (v[0])
    }
}
