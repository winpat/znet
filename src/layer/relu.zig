const std = @import("std");
const Allocator = std.mem.Allocator;
const t = std.testing;
const assert = std.debug.assert;

const Matrix = @import("../matrix.zig").Matrix;

pub fn ReLU(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        dim: usize,

        activations: Matrix(T),
        gradient: Matrix(T),

        /// Initialize relu layer.
        pub fn init(allocator: Allocator, dim: usize) !Self {
            return Self{
                .allocator = allocator,
                .dim = dim,
                .activations = try Matrix(T).alloc(allocator, 1, dim, .zeros),
                .gradient = try Matrix(T).alloc(allocator, 1, dim, .zeros),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self) void {
            self.gradient.deinit(self.allocator);
            self.activations.deinit(self.allocator);
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("ReLU(dim={} grad={})", .{
                self.dim,
                self.gradient,
            });
        }

        /// Compute the layers activation.
        pub fn forward(self: *Self, input: Matrix(T)) Matrix(T) {
            for (self.activations.elements, input.elements) |*a, z| {
                a.* = @max(0, z);
            }
            return self.activations;
        }

        /// Compute gradient given upstream gradient of the followup layers.
        pub fn backward(self: Self, input: Matrix(T), err_grad: Matrix(T)) Matrix(T) {
            assert(input.sameDimAs(err_grad));
            assert(input.sameDimAs(self.gradient));
            for (self.gradient.elements, input.elements, err_grad.elements) |*g, z, e| {
                g.* = if (z <= 0) 0 else e;
            }
            return self.gradient;
        }
    };
}

test "ReLU forward pass" {
    var input_data = [_]f32{ -0.4, 0.0, 0.3 };
    const input = Matrix(f32).init(1, 3, &input_data);

    var relu = try ReLU(f32).init(t.allocator, 3);
    defer relu.deinit();

    const prediction = relu.forward(input);

    try t.expectEqualSlices(f32, prediction.elements, &.{ 0.0, 0.0, 0.3 });
}

test "ReLU backward pass" {
    var input_data = [_]f32{ -0.4, 0.0, 0.3 };
    const input = Matrix(f32).init(1, 3, &input_data);

    var err_grad_data = [_]f32{ 0.5, 0.5, 0.5 };
    const err_grad = Matrix(f32).init(1, 3, &err_grad_data);

    const relu = try ReLU(f32).init(t.allocator, 3);
    defer relu.deinit();

    const grad = relu.backward(input, err_grad);

    try t.expectEqualSlices(f32, grad.elements, &.{ 0.0, 0.0, 0.5 });
}
