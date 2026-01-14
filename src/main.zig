const std = @import("std");
const mem = std.mem;

const accuracy = @import("score.zig").accuracy;
const cli = @import("cli.zig");
const iris = @import("iris.zig");
const Matrix = @import("matrix.zig").Matrix;
const minMaxNormalize = @import("scale.zig").minMaxNormalize;
const Network = @import("net.zig").Network;
const ops = @import("ops.zig");
const trainTestSplit = @import("split.zig").trainTestSplit;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const app = cli.App{
        .name = "iris",
        .description = "Tool for training neural networks on the iris dataset.",
        .version = "0.0.1",
        .root = .{
            .cmds = &.{
                .{
                    .name = "train",
                    .description = "Train a model.",
                    .args = &.{
                        .{
                            .name = "path",
                            .description = "Path to a CSV containing the iris dataset.",
                            .default = .{ .string = "data/risi.csv" },
                        },
                    },
                    .opts = &.{
                        .{
                            .name = "epoch",
                            .long = "epochs",
                            .description = "Number of epochs to train.",
                            .kind = .int,
                            .default = .{ .int = 30 },
                        },
                        .{
                            .name = "learning-rate",
                            .long = "learning-rate",
                            .short = "lr",
                            .description = "Learning rate.",
                            .kind = .float,
                            .default = .{ .float = 0.01 },
                        },
                    },
                },
            },
        },
    };

    const ctx = try cli.parse(allocator, &app);

    if (ctx.path.items.len == 1) {
        ctx.help();
        return;
    }

    if (mem.eql(u8, ctx.path.items[1].name, "train"))
        return try trainModel(ctx);

    unreachable;
}

pub fn trainModel(ctx: cli.Context) !void {
    const csv_path = ctx.args.get("path").?.string;
    const epochs: usize = @intCast(ctx.opts.get("epoch").?.int);
    const learning_rate: f32 = @floatCast(ctx.opts.get("learning-rate").?.float);

    try ctx.stdout.print(
        "Training models with epochs={d} and lr={d}.\n\n",
        .{ epochs, learning_rate },
    );

    var features, var labels = try iris.load(ctx.allocator, csv_path);
    minMaxNormalize(f32, &features);

    const X_train, const y_train, const X_test, const y_test = trainTestSplit(
        f32,
        &features,
        &labels,
        0.8,
        0.2,
    );

    var net = Network(f32).init(ctx.allocator, 4, 3);
    defer net.deinit();

    try net.addLinear(8);
    try net.addReLU();
    try net.addLinear(3);
    try net.addSoftmax();

    try net.train(
        epochs,
        learning_rate,
        X_train,
        y_train,
        ctx.stdout,
    );

    const pred = try net.predictBatch(X_test);
    const acc = accuracy(f32, pred, y_test);

    try ctx.stdout.print("\nAccuracy: {d:.3}\n", .{acc});
    try ctx.stdout.flush();
}
