const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const tst = std.testing;

const mat = @import("matrix.zig");
const Matrix = mat.Matrix;

/// Compute the accuracy score.
pub fn accuracy(comptime T: type, predictions: Matrix(T), labels: Matrix(T)) f32 {
    assert(mem.eql(usize, &predictions.shape, &labels.shape));

    var preds = predictions;
    var lbls = labels;
    var true_positives: f32 = 0;

    for (0..predictions.shape[0]) |r| {
        const pred = mat.row(T, &preds, r);
        const gt = mat.row(T, &lbls, r);
        if (mat.argmax(T, pred) == mat.argmax(T, gt))
            true_positives += 1;
    }

    return true_positives / @as(T, @floatFromInt(predictions.shape[0]));
}

test "Compute accuracy score" {
    var prediction_data = [_]f32{
        0.9, 0.05, 0.05,
        0.0, 1.0,  0.0,
        0.1, 0.6,  1.3,
        0.4, 0.6,  0.0, // False prediction
    };
    const predictions = Matrix(f32).fromBuffer(.{ 4, 3 }, &prediction_data);

    var label_data = [_]f32{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
    };
    const labels = Matrix(f32).fromBuffer(.{ 4, 3 }, &label_data);

    const acc = accuracy(f32, predictions, labels);
    try tst.expectEqual(@as(f32, 0.75), acc);
}
