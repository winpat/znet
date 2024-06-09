const std = @import("std");
const iris = @import("iris.zig");
const Network = @import("net.zig").Network;
const Matrix = @import("matrix.zig").Matrix;
const minMaxNormalize = @import("scale.zig").minMaxNormalize;
const accuracy = @import("score.zig").accuracy;
const ops = @import("ops.zig");
const trainTestSplit = @import("split.zig").trainTestSplit;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var features, var labels = try iris.load(allocator, "data/iris.csv");
    minMaxNormalize(f32, &features);

    const train_features, const train_labels, const test_features, const test_labels = trainTestSplit(
        f32,
        &features,
        &labels,
        0.8,
        0.2,
    );

    var net = Network(f32).init(allocator, 4, 3);
    defer net.deinit();

    try net.addLinear(8);
    try net.addReLU();
    try net.addLinear(3);
    try net.addSoftmax();

    try net.train(300, 0.01, train_features, train_labels);

    var predictions = try Matrix(f32).alloc(
        allocator,
        test_features.rows,
        test_labels.columns,
        .zeros,
    );

    for (0..test_features.rows) |r| {
        const X = test_features.getRow(r);
        const p = net.predict(X);

        // Encode predictions so that the most likely class gets mapped to 0 and
        // all the other columns to 0.
        _, const c = ops.argmax(f32, p);
        predictions.set(r, c, 1.0);
    }

    const acc = accuracy(f32, predictions, test_labels);
    std.debug.print("Accuracy: {d:.3}\n", .{acc});
}

test {
    std.testing.refAllDecls(@This());
}
