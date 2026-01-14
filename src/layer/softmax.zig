const std = @import("std");
const Allocator = std.mem.Allocator;
const t = std.testing;
const assert = std.debug.assert;

const Matrix = @import("../matrix.zig").Matrix;
const ops = @import("../ops.zig");

pub fn Softmax(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        dim: usize,

        activations: Matrix(T),
        gradient: Matrix(T),
        jacobian: Matrix(T),

        /// Initialize softmax layer.
        pub fn init(allocator: Allocator, dim: usize) !Self {
            return Self{
                .allocator = allocator,
                .dim = dim,
                .activations = try Matrix(T).alloc(allocator, 1, dim, .zeros),
                .gradient = try Matrix(T).alloc(allocator, 1, dim, .zeros),
                .jacobian = try Matrix(T).alloc(allocator, dim, dim, .zeros),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self) void {
            self.gradient.deinit(self.allocator);
            self.jacobian.deinit(self.allocator);
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
            try writer.print("Softmax(dim={} grad={})", .{
                self.dim,
                self.gradient,
            });
        }

        /// Compute layers activation.
        pub fn forward(self: *Self, input: Matrix(T)) Matrix(T) {
            var sum: T = 0;
            for (input.elements) |e| sum += @exp(e);

            for (self.activations.elements, input.elements) |*a, e| {
                a.* = @exp(e) / sum;
            }
            return self.activations;
        }

        /// Compute gradient given upstream gradient of followup layers.
        pub fn backward(self: *Self, err_grad: Matrix(T)) Matrix(T) {
            assert(self.activations.sameDimAs(err_grad));

            for (0.., self.activations.elements) |i, x| {
                for (0.., self.activations.elements) |j, y| {
                    const v = if (i == j) x * (1 - x) else -x * y;
                    self.jacobian.set(i, j, v);
                }
            }
            ops.multiply(T, err_grad, self.jacobian, &self.gradient);
            return self.gradient;
        }
    };
}

test "Softmax forward pass" {
    var input_data = [_]f32{ 1, 2 };
    const input = Matrix(f32).fromSlice(1, 2, &input_data);

    var softmax = try Softmax(f32).init(t.allocator, 2);
    defer softmax.deinit();

    const prediction = softmax.forward(input);

    try t.expectEqualSlices(
        f32,
        &.{ 2.689414e-1, 7.310586e-1 },
        prediction.elements,
    );
}

test "Softmax backward pass" {
    var input_data = [_]f32{ 1, 2 };
    const input = Matrix(f32).fromSlice(1, 2, &input_data);

    var err_grad_data = [_]f32{ -0.5, 0.5 };
    const err_grad = Matrix(f32).fromSlice(1, 2, &err_grad_data);

    var softmax = try Softmax(f32).init(t.allocator, 2);
    defer softmax.deinit();

    _ = softmax.forward(input);
    const grad = softmax.backward(err_grad);
    try t.expectEqualSlices(
        f32,
        &.{ -0.19661193, 0.19661193 },
        grad.elements,
    );
}
