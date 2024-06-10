const std = @import("std");
const assert = std.debug.assert;
const Matrix = @import("matrix.zig").Matrix;
const mem = std.mem;
const t = std.testing;
const ops = @import("ops.zig");

/// Compute the accuracy score.
pub fn accuracy(comptime T: type, predictions: Matrix(T), labels: Matrix(T)) f32 {
    assert(predictions.sameDimAs(labels));

    var true_positives: f32 = 0;

    for (0..predictions.rows) |r| {
        const p = predictions.getRow(r);
        const l = labels.getRow(r);
        if (ops.argmax(T, p) == ops.argmax(T, l)) {
            true_positives += 1;
        }
    }

    return true_positives / @as(T, @floatFromInt(predictions.rows));
}

test "Compute accuracy score" {
    var prediction_data = [_]f32{
        0.9, 0.05, 0.05,
        0.0, 1.0,  0.0,
        0.1, 0.6,  1.3,
        0.4, 0.6, 0.0, // False prediction
    };
    const predictions = Matrix(f32).init(4, 3, &prediction_data);

    var label_data = [_]f32{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
    };
    const labels = Matrix(f32).init(4, 3, &label_data);

    const acc = accuracy(f32, predictions, labels);
    try t.expectEqual(@as(f32, 0.75), acc);
}
