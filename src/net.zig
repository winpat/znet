const std = @import("std");
const tst = std.testing;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const mem = std.mem;

const mat = @import("matrix.zig");
const Matrix = mat.Matrix;
const MeanSquaredError = @import("mse.zig").MeanSquaredError;

pub fn Network(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        layers: std.ArrayList(Layer(T)) = .{},

        inputs: usize,
        outputs: usize,

        /// Initialize network.
        pub fn init(allocator: Allocator, inputs: usize, outputs: usize) Self {
            return Self{
                .allocator = allocator,
                .inputs = inputs,
                .outputs = outputs,
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: *Self) void {
            for (self.layers.items) |*layer| {
                layer.deinit(self.allocator);
            }
            self.layers.deinit(self.allocator);
        }

        /// Add a layer to the network.
        pub fn addLayer(self: *Self, layer: Layer(T)) !void {
            try self.layers.append(self.allocator, layer);
        }

        /// Return number of nodes in the last layer. If the network does not
        /// have any layer the number of inputs is returned.
        fn numNeuronsOfLastLayer(self: Self) usize {
            return if (self.layers.items.len > 0)
                self.layers.getLast().getNumOutputs()
            else
                self.inputs;
        }

        /// Add sigmoid layer to the network.
        pub fn addSigmoid(self: *Self) !void {
            const dim = self.numNeuronsOfLastLayer();
            const sigmoid = try Sigmoid(T).init(self.allocator, dim);
            const layer = Layer(T){ .sigmoid = sigmoid };
            try self.layers.append(self.allocator, layer);
        }

        /// Add ReLU layer to the network.
        pub fn addReLU(self: *Self) !void {
            const dim = self.numNeuronsOfLastLayer();
            const relu = try ReLU(T).init(self.allocator, dim);
            const layer = Layer(T){ .relu = relu };
            try self.layers.append(self.allocator, layer);
        }

        /// Add softmax layer to the network.
        pub fn addSoftmax(self: *Self) !void {
            const dim = self.numNeuronsOfLastLayer();
            const softmax = try Softmax(T).init(self.allocator, dim);
            const layer = Layer(T){ .softmax = softmax };
            try self.layers.append(self.allocator, layer);
        }

        /// Add linear layer to the network.
        pub fn addLinear(self: *Self, outputs: usize) !void {
            const inputs = self.numNeuronsOfLastLayer();
            const linear = try Linear(T).rand(self.allocator, inputs, outputs);
            const layer = Layer(T){ .linear = linear };
            try self.layers.append(self.allocator, layer);
        }

        /// Feed a single input through network.
        pub fn predict(self: Self, input: Matrix(T)) Matrix(T) {
            var state = input;
            for (self.layers.items) |*layer|
                state = layer.forward(state);
            return state;
        }

        /// Feed a batch of inputs through the network.
        pub fn predictBatch(self: Self, batch: Matrix(T)) !Matrix(T) {
            var b = batch;
            var predictions = try Matrix(T).zeros(self.allocator, .{ batch.shape[0], self.outputs });
            for (0..batch.shape[0]) |r| {
                const prediction = self.predict(mat.row(T, &b, r));
                mat.setRow(T, &predictions, r, prediction);
            }
            return predictions;
        }

        /// Propagate gradient through layers and adjust parameters.
        pub fn backward(self: Self, input: Matrix(T), grad: Matrix(T), learning_rate: f32) void {
            var err_grad = grad;
            var i = self.layers.items.len;
            while (i > 0) : (i -= 1) {
                var layer = self.layers.items[i - 1];

                err_grad = if (i > 1)
                    layer.backward(self.layers.items[i - 2].activation(), err_grad)
                else
                    layer.backward(input, err_grad);

                if (layer == .linear)
                    layer.linear.applyGradients(learning_rate);
            }
        }

        /// Train the network for fixed number of epochs.
        pub fn train(self: Self, epochs: usize, learning_rate: f32, input: Matrix(T), labels: Matrix(T), writer: *std.Io.Writer) !void {
            assert(input.shape[0] == labels.shape[0]);
            assert(input.shape[1] == self.inputs);
            assert(labels.shape[1] == self.outputs);

            // TODO Make loss function configurable
            var cost_fn = try MeanSquaredError(f32).init(self.allocator, self.outputs);
            defer cost_fn.deinit();

            const start = std.time.milliTimestamp();

            var inp = input;
            var lbls = labels;
            const num_samples = input.shape[0];
            for (0..epochs) |e| {
                var loss_per_epoch: f32 = 0;

                for (0..num_samples) |r| {
                    const X = mat.row(T, &inp, r);
                    const y = mat.row(T, &lbls, r);
                    const prediction = self.predict(X);
                    loss_per_epoch += cost_fn.computeLoss(prediction, y);

                    const err_grad = cost_fn.computeGradient(prediction, y);
                    self.backward(X, err_grad, learning_rate);
                }

                const avg_loss_per_epoch = loss_per_epoch / @as(f32, @floatFromInt(num_samples));
                try writer.print("Average loss epoch {d}: {d:.4}\n", .{ e, avg_loss_per_epoch });
                try writer.flush();
            }

            const elapsed_seconds: f32 = @as(f32, @floatFromInt(std.time.milliTimestamp() - start)) / 1000;
            try writer.print("Training took {d:.2} seconds.\n", .{elapsed_seconds});
            try writer.flush();
        }
    };
}

test "Make prediction given inputs" {
    var net = Network(f32).init(tst.allocator, 2, 4);
    defer net.deinit();

    const l1_weights = [_]f32{
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    };
    const l1_biases = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const l1 = try Linear(f32).init(tst.allocator, 2, 4, &l1_weights, &l1_biases);

    const s1 = try Sigmoid(f32).init(tst.allocator, 4);

    try net.addLayer(Layer(f32){ .linear = l1 });
    try net.addLayer(Layer(f32){ .sigmoid = s1 });

    var input_data = [_]f32{
        1.0, 1.0,
        1.0, 1.0,
    };
    const input = Matrix(f32).fromBuffer(.{ 2, 2 }, &input_data);

    const prediction = try net.predictBatch(input);
    defer prediction.deinit(tst.allocator);

    var pred = prediction;
    try tst.expectEqualSlices(
        f32,
        mat.row(f32, &pred, 0).elements,
        &.{
            9.5257413e-1, 9.5257413e-1,
            9.5257413e-1, 9.5257413e-1,
        },
    );

    try tst.expectEqualSlices(
        f32,
        mat.row(f32, &pred, 1).elements,
        &.{
            9.5257413e-1, 9.5257413e-1,
            9.5257413e-1, 9.5257413e-1,
        },
    );
}

test "Train network" {
    var net = Network(f32).init(tst.allocator, 2, 4);
    defer net.deinit();

    const l1_weights = [_]f32{
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    };
    const l1_biases = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const l1 = try Linear(f32).init(tst.allocator, 2, 4, &l1_weights, &l1_biases);

    const s1 = try Sigmoid(f32).init(tst.allocator, 4);

    try net.addLayer(Layer(f32){ .linear = l1 });
    try net.addLayer(Layer(f32){ .sigmoid = s1 });

    var input_data = [_]f32{ 1.0, 1.0 };
    const input = Matrix(f32).fromBuffer(.{ 1, 2 }, &input_data);

    var labels_data = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const labels = Matrix(f32).fromBuffer(.{ 1, 4 }, &labels_data);

    var discarding = std.Io.Writer.Discarding.init(&.{});
    try net.train(40, 0.001, input, labels, &discarding.writer);
}

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
                .linear => |layer| layer.activations.shape[1],
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
    const input = Matrix(f32).fromBuffer(.{ 1, 3 }, &input_data);

    const sigmoid = try Sigmoid(f32).init(tst.allocator, 3);
    var layer = Layer(f32){ .sigmoid = sigmoid };
    defer layer.deinit(tst.allocator);

    const prediction = layer.forward(input);

    try tst.expectEqualSlices(f32, prediction.elements, &.{ 7.310586e-1, 8.80797e-1, 9.5257413e-1 });
}

test "Backward pass" {
    var input_data = [_]f32{ 1.0, 2.0, 3.0 };
    const input = Matrix(f32).fromBuffer(.{ 1, 3 }, &input_data);

    var err_grad_data = [_]f32{ 0.5, 0.5, 0.5 };
    const err_grad = Matrix(f32).fromBuffer(.{ 1, 3 }, &err_grad_data);

    const sigmoid = try Sigmoid(f32).init(tst.allocator, 3);
    var layer = Layer(f32){ .sigmoid = sigmoid };
    defer layer.deinit(tst.allocator);

    _ = layer.forward(input);
    const grad = layer.backward(input, err_grad);

    try tst.expectEqualSlices(f32, grad.elements, &.{ 9.830596e-2, 5.2496813e-2, 2.2588328e-2 });
}

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
                .activations = try Matrix(T).zeros(allocator, .{ 1, outputs }),
                .weights = try Matrix(T).fromSlice(allocator, .{ inputs, outputs }, weights),
                .weights_grad = try Matrix(T).zeros(allocator, .{ inputs, outputs }),
                .weights_t = try Matrix(T).fromSlice(allocator, .{ outputs, inputs }, weights),
                .biases = try Matrix(T).fromSlice(allocator, .{ 1, outputs }, biases),
                .biases_grad = try Matrix(T).zeros(allocator, .{ 1, outputs }),
                .inputs_grad = try Matrix(T).zeros(allocator, .{ 1, inputs }),
                .inputs_t = try Matrix(T).zeros(allocator, .{ inputs, 1 }),
            };
        }

        /// Randomly initalize linear layer.
        pub fn rand(allocator: Allocator, inputs: usize, outputs: usize) !Self {
            return Self{
                .activations = try Matrix(T).zeros(allocator, .{ 1, outputs }),
                .weights = try Matrix(T).rand(allocator, .{ inputs, outputs }),
                .weights_grad = try Matrix(T).zeros(allocator, .{ inputs, outputs }),
                .weights_t = try Matrix(T).rand(allocator, .{ outputs, inputs }),
                .biases = try Matrix(T).rand(allocator, .{ 1, outputs }),
                .biases_grad = try Matrix(T).zeros(allocator, .{ 1, outputs }),
                .inputs_grad = try Matrix(T).zeros(allocator, .{ 1, inputs }),
                .inputs_t = try Matrix(T).zeros(allocator, .{ inputs, 1 }),
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
            mat.mul(T, input, self.weights, &self.activations);
            mat.add(T, self.activations, self.biases, &self.activations);
            return self.activations;
        }

        /// Compute input, weight and bias gradients given upstream gradient of
        /// the followup layers.
        pub fn backward(self: *Self, input: Matrix(T), err_grad: Matrix(T)) Matrix(T) {
            // dC/db = err_grad
            @memcpy(self.biases_grad.elements, err_grad.elements);

            // dC/dw = input^T @ err_grad
            mat.transpose(T, input, &self.inputs_t);
            mat.mul(T, self.inputs_t, err_grad, &self.weights_grad);

            // dC/di = err_grad @ weights^T
            mat.transpose(T, self.weights, &self.weights_t);
            mat.mul(T, err_grad, self.weights_t, &self.inputs_grad);

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
    const features = Matrix(f32).fromBuffer(.{ 1, 2 }, &feature_data);

    // 2x4 [ 1.0  1.0  1.0  1.0
    //       1.0  1.0  1.0  1.0 ]
    const weights = &.{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    // 1x4 [ 1.0  1.0  1.0  1.0 ]
    const biases = &.{ 1.0, 1.0, 1.0, 1.0 };

    var layer = try Linear(f32).init(tst.allocator, 2, 4, weights, biases);
    defer layer.deinit(tst.allocator);

    const prediction = layer.forward(features);

    try tst.expectEqualSlices(f32, prediction.elements, &.{ 3.0, 3.0, 3.0, 3.0 });
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

    var linear = try Linear(f32).init(tst.allocator, 4, 6, &weights, &biases);
    defer linear.deinit(tst.allocator);

    // The first row of the iris dataset.
    const features = [_]f32{ 5.1, 3.5, 1.4, 0.2 };
    const input = try Matrix(f32).fromSlice(tst.allocator, .{ 1, 4 }, &features);
    defer input.deinit(tst.allocator);

    const err_grad_data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    var err_grad = try Matrix(f32).fromSlice(tst.allocator, .{ 1, 6 }, &err_grad_data);
    defer err_grad.deinit(tst.allocator);

    const grad = linear.backward(input, err_grad);

    try tst.expectEqualSlices(f32, grad.elements, &.{ 3.1999998e-2, 6.0000002e-2, 9.5e-2, 1.095 });
    try tst.expectEqualSlices(f32, linear.biases_grad.elements, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    try tst.expectEqualSlices(f32, linear.weights_grad.elements, &.{
        5.1e-1,       1.02e0,       1.5300001e0,  2.04e0,       2.55e0, 3.0600002e0,
        3.5e-1,       7e-1,         1.0500001e0,  1.4e0,        1.75e0, 2.1000001e0,
        1.4e-1,       2.8e-1,       4.2000002e-1, 5.6e-1,       7e-1,   8.4000003e-1,
        2.0000001e-2, 4.0000003e-2, 6.0000002e-2, 8.0000006e-2, 1e-1,   1.20000005e-1,
    });
}

pub fn ReLU(comptime T: type) type {
    return struct {
        const Self = @This();

        dim: usize,

        activations: Matrix(T),
        gradient: Matrix(T),

        /// Initialize relu layer.
        pub fn init(allocator: Allocator, dim: usize) !Self {
            return Self{
                .dim = dim,
                .activations = try Matrix(T).zeros(allocator, .{ 1, dim }),
                .gradient = try Matrix(T).zeros(allocator, .{ 1, dim }),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self, allocator: Allocator) void {
            self.gradient.deinit(allocator);
            self.activations.deinit(allocator);
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
            assert(mem.eql(usize, &input.shape, &err_grad.shape));
            assert(mem.eql(usize, &input.shape, &self.gradient.shape));
            for (self.gradient.elements, input.elements, err_grad.elements) |*g, z, e| {
                g.* = if (z <= 0) 0 else e;
            }
            return self.gradient;
        }
    };
}

test "ReLU forward pass" {
    var input_data = [_]f32{ -0.4, 0.0, 0.3 };
    const input = Matrix(f32).fromBuffer(.{ 1, 3 }, &input_data);

    var relu = try ReLU(f32).init(tst.allocator, 3);
    defer relu.deinit(tst.allocator);

    const prediction = relu.forward(input);

    try tst.expectEqualSlices(f32, prediction.elements, &.{ 0.0, 0.0, 0.3 });
}

test "ReLU backward pass" {
    var input_data = [_]f32{ -0.4, 0.0, 0.3 };
    const input = Matrix(f32).fromBuffer(.{ 1, 3 }, &input_data);

    var err_grad_data = [_]f32{ 0.5, 0.5, 0.5 };
    const err_grad = Matrix(f32).fromBuffer(.{ 1, 3 }, &err_grad_data);

    const relu = try ReLU(f32).init(tst.allocator, 3);
    defer relu.deinit(tst.allocator);

    const grad = relu.backward(input, err_grad);

    try tst.expectEqualSlices(f32, grad.elements, &.{ 0.0, 0.0, 0.5 });
}

pub fn Softmax(comptime T: type) type {
    return struct {
        const Self = @This();

        dim: usize,

        activations: Matrix(T),
        gradient: Matrix(T),
        jacobian: Matrix(T),

        /// Initialize softmax layer.
        pub fn init(allocator: Allocator, dim: usize) !Self {
            return Self{
                .dim = dim,
                .activations = try Matrix(T).zeros(allocator, .{ 1, dim }),
                .gradient = try Matrix(T).zeros(allocator, .{ 1, dim }),
                .jacobian = try Matrix(T).zeros(allocator, .{ dim, dim }),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self, allocator: Allocator) void {
            self.gradient.deinit(allocator);
            self.jacobian.deinit(allocator);
            self.activations.deinit(allocator);
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
            assert(mem.eql(usize, &self.activations.shape, &err_grad.shape));

            for (0.., self.activations.elements) |i, x| {
                for (0.., self.activations.elements) |j, y| {
                    const v = if (i == j) x * (1 - x) else -x * y;
                    self.jacobian.set(.{ i, j }, v);
                }
            }
            mat.mul(T, err_grad, self.jacobian, &self.gradient);
            return self.gradient;
        }
    };
}

test "Softmax forward pass" {
    var input_data = [_]f32{ 1, 2 };
    const input = Matrix(f32).fromBuffer(.{ 1, 2 }, &input_data);

    var softmax = try Softmax(f32).init(tst.allocator, 2);
    defer softmax.deinit(tst.allocator);

    const prediction = softmax.forward(input);

    try tst.expectEqualSlices(
        f32,
        &.{ 2.689414e-1, 7.310586e-1 },
        prediction.elements,
    );
}

test "Softmax backward pass" {
    var input_data = [_]f32{ 1, 2 };
    const input = Matrix(f32).fromBuffer(.{ 1, 2 }, &input_data);

    var err_grad_data = [_]f32{ -0.5, 0.5 };
    const err_grad = Matrix(f32).fromBuffer(.{ 1, 2 }, &err_grad_data);

    var softmax = try Softmax(f32).init(tst.allocator, 2);
    defer softmax.deinit(tst.allocator);

    _ = softmax.forward(input);
    const grad = softmax.backward(err_grad);
    try tst.expectEqualSlices(
        f32,
        &.{ -0.19661193, 0.19661193 },
        grad.elements,
    );
}

pub fn Sigmoid(comptime T: type) type {
    return struct {
        const Self = @This();

        dim: usize,

        activations: Matrix(T),
        gradient: Matrix(T),

        /// Initalize sigmoid layer
        pub fn init(allocator: Allocator, dim: usize) !Self {
            return Self{
                .dim = dim,
                .activations = try Matrix(T).zeros(allocator, .{ 1, dim }),
                .gradient = try Matrix(T).zeros(allocator, .{ 1, dim }),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self, allocator: Allocator) void {
            self.gradient.deinit(allocator);
            self.activations.deinit(allocator);
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
    const input = Matrix(f32).fromBuffer(.{ 1, 3 }, &input_data);

    var sigmoid = try Sigmoid(f32).init(tst.allocator, 3);
    defer sigmoid.deinit(tst.allocator);

    const prediction = sigmoid.forward(input);

    try tst.expectEqualSlices(f32, prediction.elements, &.{ 7.310586e-1, 8.80797e-1, 9.5257413e-1 });
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
    const input = Matrix(f32).fromBuffer(.{ 1, 3 }, &input_data);

    var err_grad_data = [_]f32{ 0.5, 0.5, 0.5 };
    const err_grad = Matrix(f32).fromBuffer(.{ 1, 3 }, &err_grad_data);

    var sigmoid = try Sigmoid(f32).init(tst.allocator, 3);
    defer sigmoid.deinit(tst.allocator);

    _ = sigmoid.forward(input);
    const grad = sigmoid.backward(err_grad);

    try tst.expectEqualSlices(f32, grad.elements, &.{ 9.830596e-2, 5.2496813e-2, 2.2588328e-2 });
}
