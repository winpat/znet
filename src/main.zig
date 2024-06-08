const std = @import("std");
const iris = @import("iris.zig");
const Network = @import("net.zig").Network;
const Matrix = @import("matrix.zig").Matrix;
const minMaxNormalize = @import("scale.zig").minMaxNormalize;
const accuracy = @import("score.zig").accuracy;
const ops = @import("ops.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var features, const labels = try iris.load(
        allocator,
        "data/iris.csv",
    );

    minMaxNormalize(f32, &features);

    var net = Network(f32).init(allocator, 4, 3);
    defer net.deinit();

    try net.addLinear(8);
    try net.addReLU();
    try net.addLinear(3);
    try net.addSoftmax();

    try net.train(300, 0.01, features, labels);

    var predictions = try Matrix(f32).alloc(allocator, features.rows, labels.columns, .zeros);

    for (0..features.rows) |r| {
        const X = features.getRow(r);
        const p = net.predict(X);

        // Encode predictions so that the most likely class gets mapped to 0 and
        // all the other columns to 0.
        _, const c = ops.argmax(f32, p);
        predictions.set(r, c, 1.0);
    }

    const acc = accuracy(f32, predictions, labels);
    std.debug.print("Accuracy: {d:.3}\n", .{acc});
}

test {
    std.testing.refAllDecls(@This());
}
