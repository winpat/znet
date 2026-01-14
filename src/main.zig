const std = @import("std");

const accuracy = @import("score.zig").accuracy;
const iris = @import("iris.zig");
const Matrix = @import("matrix.zig").Matrix;
const minMaxNormalize = @import("scale.zig").minMaxNormalize;
const Network = @import("net.zig").Network;
const ops = @import("ops.zig");
const trainTestSplit = @import("split.zig").trainTestSplit;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var buf = [_]u8{0} ** 1024;
    var stdout_writer = std.fs.File.stdout().writer(&buf);
    const stdout = &stdout_writer.interface;

    var features, var labels = try iris.load(allocator, "data/iris.csv");
    minMaxNormalize(f32, &features);

    const X_train, const y_train, const X_test, const y_test = trainTestSplit(
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

    try net.train(300, 0.01, 32, X_train, y_train, stdout);

    const predictions = try net.predictBatch(X_test);
    const acc = accuracy(f32, predictions, y_test);
    std.debug.print("Accuracy: {d:.3}\n", .{acc});
}

test {
    std.testing.refAllDecls(@This());
}
