// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// `leftPadding` and `kvOffsets` must give the same attention as the equivalent explicit
/// boolean mask (PicoMLX/mlx fork PRs 8 and 10).
class MLXFastLeftPaddingTests: XCTestCase {

    private func inputs(
        batch: Int, queryHeads: Int, kvHeads: Int, queries: Int, keys: Int, dim: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(11)
        let q = MLXRandom.normal([batch, queryHeads, queries, dim])
        let k = MLXRandom.normal([batch, kvHeads, keys, dim])
        let v = MLXRandom.normal([batch, kvHeads, keys, dim])
        return (q, k, v)
    }

    /// `[B, 1, T_q, T_kv]` mask: row b, query j sees keys `pads[b] ... ends[b] + j`.
    private func explicitMask(pads: [Int32], ends: [Int32], queries: Int, keys: Int) -> MLXArray {
        let key = MLXArray(Int32(0) ..< Int32(keys)).reshaped([1, 1, 1, keys])
        let start = MLXArray(pads).reshaped([pads.count, 1, 1, 1])
        let last =
            MLXArray(ends).reshaped([ends.count, 1, 1, 1])
            + MLXArray(Int32(0) ..< Int32(queries)).reshaped([1, 1, queries, 1])
        return (key .>= start) .&& (key .<= last)
    }

    func testLeftPaddingMatchesMask() {
        // 1 query (vector kernel) and 4 causal queries, short and long rows (both vector passes).
        for (queries, keys) in [(1, 300), (1, 2100), (4, 300), (4, 2100)] {
            let (q, k, v) = inputs(
                batch: 4, queryHeads: 8, kvHeads: 2, queries: queries, keys: keys, dim: 64)
            let pads: [Int32] = [0, Int32(keys / 2), Int32(keys - 40), 7]
            // One query runs without a mask mode (it sees its whole row); several queries
            // use lower-right causal masking, so query j sees keys up to keys - queries + j.
            let causal = queries > 1
            let ends = [Int32](repeating: Int32(keys - queries), count: 4)
            let mask = explicitMask(pads: pads, ends: ends, queries: queries, keys: keys)
            let scale: Float = 0.125

            let expected = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
            let actual = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale,
                leftPadding: MLXArray(pads), causal: causal)
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertTrue(
                allClose(actual, expected, rtol: 1e-4, atol: 1e-4).item(),
                "queries \(queries), keys \(keys)")
        }
    }

    func testKVOffsetsSkipStaleSuffix() {
        // Rows end at different columns; keys after each row's cursor are stale.
        let queries = 3
        let keys = 1200
        let (q, k, v) = inputs(
            batch: 3, queryHeads: 4, kvHeads: 1, queries: queries, keys: keys, dim: 64)
        let pads: [Int32] = [0, 100, 600]
        let offsets: [Int32] = [Int32(keys - queries), 400, 900]
        let mask = explicitMask(pads: pads, ends: offsets, queries: queries, keys: keys)

        let expected = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 0.125, mask: .array(mask))
        let actual = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 0.125,
            leftPadding: MLXArray(pads), causal: true, kvOffsets: MLXArray(offsets))
        XCTAssertTrue(allClose(actual, expected, rtol: 1e-4, atol: 1e-4).item())
    }
}
