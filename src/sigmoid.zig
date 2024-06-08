const std = @import("std");
const Allocator = std.mem.Allocator;
const Matrix = @import("matrix.zig").Matrix;
const t = std.testing;
const assert = std.debug.assert;

pub fn Sigmoid(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        dim: usize,

        activations: Matrix(T),
        gradient: Matrix(T),

        /// Initalize sigmoid layer
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
            self.gradient.free(self.allocator);
            self.activations.free(self.allocator);
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("Sigmoid(dim={} grad={})", .{
                self.dim,
                self.gradient,
            });
        }

        /// Compute the layers activation.
        pub fn forward(self: *Self, input: Matrix(T)) Matrix(T) {
            for (self.activations.elements, input.elements) |*a, z| {
                a.* = 1 / (1 + @exp(-z));
            }
            return self.activations;
        }

        /// Compute gradient given upstream gradient of the followup layers.
        pub fn backward(self: Self, err_grad: Matrix(T)) Matrix(T) {
            for (
                self.gradient.elements,
                self.activations.elements,
                err_grad.elements,
            ) |*g, a, e| {
                g.* = a * (1 - a) * e;
            }
            return self.gradient;
        }
    };
}

test "Sigmoid forward pass" {
    // Python verification:
    //
    //   import numpy as np
    //
    //   x = np.array([[1,2,3]])
    //
    //   def sigmoid(x):
    //       return 1 / (1 + np.exp(-x))
    //
    //   assert np.allclose(
    //       sigmoid(x),
    //       np.array([[0.73105858, 0.88079708, 0.95257413]])
    //   )

    var input_data = [_]f32{ 1.0, 2.0, 3.0 };
    const input = Matrix(f32).init(1, 3, &input_data);

    var sigmoid = try Sigmoid(f32).init(t.allocator, 3);
    defer sigmoid.deinit();

    const prediction = sigmoid.forward(input);

    try t.expectEqualSlices(f32, prediction.elements, &.{ 7.310586e-1, 8.80797e-1, 9.5257413e-1 });
}

test "Sigmoid backward pass" {
    // Python verification:
    //
    //   import numpy as np
    //
    //   x = np.array([[1, 2, 3]])
    //   grad = np.array([[0.5, 0.5, 0.5]])
    //
    //   def sigmoid(x):
    //       return 1 / (1 + np.exp(-x))
    //
    //   def sigmoid_deriv(x):
    //       a = sigmoid(x)
    //       return a * (1 - a)
    //
    //   assert np.allclose(
    //       sigmoid_deriv(x) * grad,
    //       np.array([[0.09830597, 0.05249679, 0.02258833]])
    //   )

    var input_data = [_]f32{ 1.0, 2.0, 3.0 };
    const input = Matrix(f32).init(1, 3, &input_data);

    var err_grad_data = [_]f32{ 0.5, 0.5, 0.5 };
    const err_grad = Matrix(f32).init(1, 3, &err_grad_data);

    var sigmoid = try Sigmoid(f32).init(t.allocator, 3);
    defer sigmoid.deinit();

    _ = sigmoid.forward(input);
    const grad = sigmoid.backward(err_grad);

    try t.expectEqualSlices(f32, grad.elements, &.{ 9.830596e-2, 5.2496813e-2, 2.2588328e-2 });
}
