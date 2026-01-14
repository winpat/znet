const std = @import("std");
const t = std.testing;
const Allocator = std.mem.Allocator;

pub const Linear = @import("layer/linear.zig").Linear;
pub const ReLU = @import("layer/relu.zig").ReLU;
pub const Sigmoid = @import("layer/sigmoid.zig").Sigmoid;
pub const Softmax = @import("layer/softmax.zig").Softmax;
const Matrix = @import("matrix.zig").Matrix;

const LayerTag = enum {
    linear,
    relu,
    sigmoid,
    softmax,
};

pub fn Layer(comptime T: type) type {
    return union(LayerTag) {
        const Self = @This();

        linear: Linear(T),
        relu: ReLU(T),
        sigmoid: Sigmoid(T),
        softmax: Softmax(T),

        /// Free all allocated memory.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            switch (self.*) {
                inline else => |*layer| layer.deinit(allocator),
            }
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            switch (self) {
                inline else => |layer| try writer.print("{}", .{layer}),
            }
        }

        /// Return the current activation.
        pub fn activation(self: Self) Matrix(T) {
            return switch (self) {
                inline else => |layer| layer.activations,
            };
        }

        /// Return number of output nodes.
        pub fn getNumOutputs(self: Self) usize {
            return switch (self) {
                .linear => |layer| layer.activations.columns,
                inline else => |layer| layer.dim,
            };
        }

        /// Compute layers activation.
        pub fn forward(self: *Self, input: Matrix(T)) Matrix(T) {
            return switch (self.*) {
                inline else => |*layer| layer.forward(input),
            };
        }

        /// Propagate gradient of follow up layers backwards.
        pub fn backward(self: *Self, input: Matrix(T), err_grad: Matrix(T)) Matrix(T) {
            return switch (self.*) {
                .softmax => |*layer| layer.backward(err_grad),
                .sigmoid => |*layer| layer.backward(err_grad),
                inline else => |*layer| layer.backward(input, err_grad),
            };
        }
    };
}

test "Forward pass" {
    var input_data = [_]f32{ 1.0, 2.0, 3.0 };
    const input = Matrix(f32).fromSlice(1, 3, &input_data);

    const sigmoid = try Sigmoid(f32).init(t.allocator, 3);
    var layer = Layer(f32){ .sigmoid = sigmoid };
    defer layer.deinit(t.allocator);

    const prediction = layer.forward(input);

    try t.expectEqualSlices(f32, prediction.elements, &.{ 7.310586e-1, 8.80797e-1, 9.5257413e-1 });
}

test "Backward pass" {
    var input_data = [_]f32{ 1.0, 2.0, 3.0 };
    const input = Matrix(f32).fromSlice(1, 3, &input_data);

    var err_grad_data = [_]f32{ 0.5, 0.5, 0.5 };
    const err_grad = Matrix(f32).fromSlice(1, 3, &err_grad_data);

    const sigmoid = try Sigmoid(f32).init(t.allocator, 3);
    var layer = Layer(f32){ .sigmoid = sigmoid };
    defer layer.deinit(t.allocator);

    _ = layer.forward(input);
    const grad = layer.backward(input, err_grad);

    try t.expectEqualSlices(f32, grad.elements, &.{ 9.830596e-2, 5.2496813e-2, 2.2588328e-2 });
}
