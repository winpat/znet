const std = @import("std");
const assert = std.debug.assert;
const t = std.testing;

const Matrix = @import("matrix.zig").Matrix;

/// Split features and labels into a train and test set.
pub fn trainTestSplit(
    comptime T: type,
    features: *Matrix(T),
    labels: *Matrix(T),
    train_size: f32,
    test_size: f32,
) struct { Matrix(T), Matrix(T), Matrix(T), Matrix(T) } {
    assert(test_size > 0 and test_size <= 1);
    assert(train_size > 0 and test_size <= 1);
    assert(train_size + test_size == 1);
    assert(features.rows == labels.rows);

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    var i = features.rows - 1;
    while (i > 0) : (i -= 1) {
        const new_pos = random.intRangeLessThan(usize, 0, i);
        features.swapRows(i, new_pos);
        labels.swapRows(i, new_pos);
    }

    const dividing_row: usize = @intFromFloat(@round(@as(f32, @floatFromInt(features.rows)) * train_size));
    const train_features, const test_features = features.splitOnRow(dividing_row);
    const train_labels, const test_labels = labels.splitOnRow(dividing_row);

    return .{
        train_features,
        train_labels,
        test_features,
        test_labels,
    };
}

test "Create test train split" {
    var feature_data = [_]f32{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    };
    var features = Matrix(f32).init(4, 2, &feature_data);

    var label_data = [_]f32{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    };
    var labels = Matrix(f32).init(4, 2, &label_data);

    const train_features, const train_labels, const test_features, const test_labels = trainTestSplit(
        f32,
        &features,
        &labels,
        0.75,
        0.25,
    );

    train_features.print();

    try t.expectEqual(train_features.rows, 3);
    try t.expectEqual(train_labels.rows, 3);
    try t.expectEqual(test_features.rows, 1);
    try t.expectEqual(test_labels.rows, 1);
}
