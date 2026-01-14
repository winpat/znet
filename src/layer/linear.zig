const std = @import("std");
const Allocator = std.mem.Allocator;
const t = std.testing;
const assert = std.debug.assert;

const Matrix = @import("../matrix.zig").Matrix;
const ops = @import("../ops.zig");

pub fn Linear(comptime T: type) type {
    return struct {
        const Self = @This();

        /// A 1xO matrix holding the current activations of the neurons.
        activations: Matrix(T),

        /// A IxO matrix where columns holds the weights connecting a neuron to
        /// the ones in the previous layer.
        weights: Matrix(T),
        weights_grad: Matrix(T),
        weights_t: Matrix(T),

        /// A 1xO matrix where each rows hold the bias of the layers neurons.
        biases: Matrix(T),
        biases_grad: Matrix(T),

        inputs_grad: Matrix(T),
        inputs_t: Matrix(T),

        /// Initialize linear layer from existing weights and biases.
        pub fn init(allocator: Allocator, inputs: usize, outputs: usize, weights: []const T, biases: []const T) !Self {
            assert(inputs * outputs == weights.len);
            assert(outputs == biases.len);

            return Self{
                .activations = try Matrix(T).init(allocator, 1, outputs, .zeros),
                .weights = try Matrix(T).initFromSlice(allocator, inputs, outputs, weights),
                .weights_grad = try Matrix(T).init(allocator, inputs, outputs, .zeros),
                .weights_t = try Matrix(T).initFromSlice(allocator, outputs, inputs, weights),
                .biases = try Matrix(T).initFromSlice(allocator, 1, outputs, biases),
                .biases_grad = try Matrix(T).init(allocator, 1, outputs, .zeros),
                .inputs_grad = try Matrix(T).init(allocator, 1, inputs, .zeros),
                .inputs_t = try Matrix(T).init(allocator, inputs, 1, .zeros),
            };
        }

        /// Randomly initalize linear layer.
        pub fn rand(allocator: Allocator, inputs: usize, outputs: usize) !Self {
            return Self{
                .activations = try Matrix(T).init(allocator, 1, outputs, .zeros),
                .weights = try Matrix(T).init(allocator, inputs, outputs, .rand),
                .weights_grad = try Matrix(T).init(allocator, inputs, outputs, .zeros),
                .weights_t = try Matrix(T).init(allocator, outputs, inputs, .rand),
                .biases = try Matrix(T).init(allocator, 1, outputs, .rand),
                .biases_grad = try Matrix(T).init(allocator, 1, outputs, .zeros),
                .inputs_grad = try Matrix(T).init(allocator, 1, inputs, .zeros),
                .inputs_t = try Matrix(T).init(allocator, inputs, 1, .zeros),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self, allocator: Allocator) void {
            self.activations.deinit(allocator);
            self.weights.deinit(allocator);
            self.weights_grad.deinit(allocator);
            self.weights_t.deinit(allocator);
            self.biases.deinit(allocator);
            self.biases_grad.deinit(allocator);
            self.inputs_grad.deinit(allocator);
            self.inputs_t.deinit(allocator);
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("Linear(i={} o={} g_w={} g_b={} g_i={})", .{
                self.inputs,
                self.outputs,
                self.weights_grad,
                self.biases_grad,
                self.inputs_grad,
            });
        }

        /// Compute the layers activation.
        pub fn forward(self: *Self, input: Matrix(T)) Matrix(T) {
            ops.multiply(T, input, self.weights, &self.activations);
            ops.add(T, self.activations, self.biases, &self.activations);
            return self.activations;
        }

        /// Compute input, weight and bias gradients given upstream gradient of
        /// the followup layers.
        pub fn backward(self: *Self, input: Matrix(T), err_grad: Matrix(T)) Matrix(T) {
            // dC/db = err_grad
            self.biases_grad.copy(err_grad);

            // dC/dw = input^T @ err_grad
            input.transpose(&self.inputs_t);
            self.inputs_t.multiply(err_grad, &self.weights_grad);

            // dC/di = err_grad @ weights^T
            self.weights.transpose(&self.weights_t);
            err_grad.multiply(self.weights_t, &self.inputs_grad);

            return self.inputs_grad;
        }

        // Apply weight and bias gradients to layer.
        pub fn applyGradients(self: *Self, learning_rate: f32) void {
            for (self.weights.elements, self.weights_grad.elements) |*w, g| {
                w.* += g * learning_rate;
            }
            for (self.biases.elements, self.biases_grad.elements) |*b, g| {
                b.* += g * learning_rate;
            }
        }
    };
}

test "Linear forward pass" {
    // Python verification
    //
    //   import numpy as np
    //
    //   X = np.array([[1, 1]])
    //
    //   W = np.array([[1, 1, 1, 1],
    //                 [1, 1, 1, 1]])
    //
    //   b = np.array([[1, 1, 1, 1]])
    //
    //   np.allclose(
    //       X.dot(W) + b,
    //       np.array([[3, 3, 3, 3]])
    //   )

    // 1x2 [ 1.0  1.0 ]
    var feature_data = [_]f32{1.0} ** 2;
    const features = Matrix(f32).fromSlice(1, 2, &feature_data);

    // 2x4 [ 1.0  1.0  1.0  1.0
    //       1.0  1.0  1.0  1.0 ]
    const weights = &.{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    // 1x4 [ 1.0  1.0  1.0  1.0 ]
    const biases = &.{ 1.0, 1.0, 1.0, 1.0 };

    var layer = try Linear(f32).init(t.allocator, 2, 4, weights, biases);
    defer layer.deinit(t.allocator);

    const prediction = layer.forward(features);

    try t.expectEqualSlices(f32, prediction.elements, &.{ 3.0, 3.0, 3.0, 3.0 });
}

test "Linear backward pass" {
    // Python verification:
    //
    //   import numpy as np
    //
    //   X = np.array([[5.1, 3.5, 1.4, 0.2 ]])
    //
    //   W = np.array(
    //       [[0.01, 0.01, 0.01, 0.01, 0.02, 0.02],
    //        [0.02, 0.02, 0.03, 0.03, 0.03, 0.03],
    //        [0.04, 0.04, 0.04, 0.04, 0.05, 0.05],
    //        [0.05, 0.05, 0.6,  0.6,  0.6,  0.6]]
    //   )
    //
    //   b = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    //
    //   g = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    //
    //   dC_db = g
    //   np.allclose(
    //       dC_db,
    //       np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    //   )
    //
    //   dC_dW = X.T.dot(g)
    //   np.allclose(
    //       dC_dW,
    //       np.array([[0.51, 1.02, 1.53, 2.04, 2.55, 3.06],
    //                 [0.35, 0.7 , 1.05, 1.4 , 1.75, 2.1 ],
    //                 [0.14, 0.28, 0.42, 0.56, 0.7 , 0.84],
    //                 [0.02, 0.04, 0.06, 0.08, 0.1 , 0.12]])
    //   )
    //
    //   dC_dX = g.dot(W.T)
    //   np.allclose(
    //       dC_dX,
    //       np.array([[0.032, 0.06 , 0.095, 1.095]])
    //   )

    var weights = [_]f32{
        0.01, 0.01, 0.01, 0.01, 0.02, 0.02,
        0.02, 0.02, 0.03, 0.03, 0.03, 0.03,
        0.04, 0.04, 0.04, 0.04, 0.05, 0.05,
        0.05, 0.05, 0.6,  0.6,  0.6,  0.6,
    };
    var biases = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };

    var linear = try Linear(f32).init(t.allocator, 4, 6, &weights, &biases);
    defer linear.deinit(t.allocator);

    // The first row of the iris dataset.
    const features = [_]f32{ 5.1, 3.5, 1.4, 0.2 };
    const input = try Matrix(f32).initFromSlice(t.allocator, 1, 4, &features);
    defer input.deinit(t.allocator);

    const err_grad_data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    var err_grad = try Matrix(f32).initFromSlice(t.allocator, 1, 6, &err_grad_data);
    defer err_grad.deinit(t.allocator);

    const grad = linear.backward(input, err_grad);

    try t.expectEqualSlices(f32, grad.elements, &.{ 3.1999998e-2, 6.0000002e-2, 9.5e-2, 1.095 });
    try t.expectEqualSlices(f32, linear.biases_grad.elements, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    try t.expectEqualSlices(f32, linear.weights_grad.elements, &.{
        5.1e-1,       1.02e0,       1.5300001e0,  2.04e0,       2.55e0, 3.0600002e0,
        3.5e-1,       7e-1,         1.0500001e0,  1.4e0,        1.75e0, 2.1000001e0,
        1.4e-1,       2.8e-1,       4.2000002e-1, 5.6e-1,       7e-1,   8.4000003e-1,
        2.0000001e-2, 4.0000003e-2, 6.0000002e-2, 8.0000006e-2, 1e-1,   1.20000005e-1,
    });
}
