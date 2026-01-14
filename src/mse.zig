const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;
const assert = std.debug.assert;
const t = std.testing;

const Matrix = @import("matrix.zig").Matrix;

pub fn MeanSquaredError(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        dim: usize,
        gradient: Matrix(T),

        /// Initialize mean squarred error cost function.
        pub fn init(allocator: Allocator, dim: usize) !Self {
            return Self{
                .allocator = allocator,
                .dim = dim,
                .gradient = try Matrix(T).init(allocator, 1, dim, .zeros),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self) void {
            self.gradient.deinit(self.allocator);
        }

        /// Compute loss between prediction and labels.
        pub fn computeLoss(self: Self, prediction: Matrix(T), labels: Matrix(T)) T {
            assert(prediction.sameDimAs(labels));

            var sum: T = 0;
            for (prediction.elements, labels.elements) |x, y| {
                sum += math.pow(T, y - x, 2);
            }
            return sum / @as(T, @floatFromInt(self.dim));
        }

        /// Compute partial derivative of the error with respect to the input.
        pub fn computeGradient(self: *Self, prediction: Matrix(T), labels: Matrix(T)) Matrix(T) {
            assert(prediction.sameDimAs(labels));

            for (self.gradient.elements, prediction.elements, labels.elements) |*g, x, y| {
                g.* = 2 * (y - x);
            }

            return self.gradient;
        }
    };
}

test "Compute the loss" {
    // Python verification:
    //
    //   import numpy as np
    //
    //   a = np.array([[1.0, 2.0, 3.0]])
    //   b = np.array([[0.0, 1.0, 0.0]])
    //
    //   def mse(x, y):
    //       return np.square(y - x).mean()
    //
    //   assert mse(a,b) == 11 / 3

    var a_data = [_]f32{ 1.0, 2.0, 3.0 };
    const a = Matrix(f32).fromSlice(1, 3, &a_data);

    var b_data = [_]f32{ 0.0, 1.0, 0.0 };
    const b = Matrix(f32).fromSlice(1, 3, &b_data);

    const mse = try MeanSquaredError(f32).init(t.allocator, 3);
    defer mse.deinit();

    const loss = mse.computeLoss(a, b);

    try t.expectEqual(loss, 11.0 / 3.0);
}

test "Compute gradient of the loss with respect to the input" {
    // Python verification:
    //
    //   import numpy as np
    //
    //   a = np.array([[1.0, 2.0, 3.0]])
    //   b = np.array([[0.0, 1.0, 0.0]])
    //
    //   def mse_deriv(x, y):
    //       return 2 * (y - x)
    //
    //   np.allclose(
    //       mse_deriv(a,b),
    //       np.array([[-2, -2, -6]])
    //   )

    var a_data = [_]f32{ 1.0, 2.0, 3.0 };
    const a = Matrix(f32).fromSlice(1, 3, &a_data);

    var b_data = [_]f32{ 0.0, 1.0, 0.0 };
    const b = Matrix(f32).fromSlice(1, 3, &b_data);

    var mse = try MeanSquaredError(f32).init(t.allocator, 3);
    defer mse.deinit();

    _ = mse.computeLoss(a, b);
    const grad = mse.computeGradient(a, b);

    try t.expectEqualSlices(f32, grad.elements, &.{ -2.0, -2.0, -6.0 });
}
