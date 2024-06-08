const Allocator = std.mem.Allocator;
const Matrix = @import("matrix.zig").Matrix;
const MeanSquaredError = @import("mse.zig").MeanSquaredError;
const Layer = @import("layer.zig").Layer;
const Linear = @import("layer.zig").Linear;
const Sigmoid = @import("layer.zig").Sigmoid;
const ReLU = @import("layer.zig").ReLU;
const Softmax = @import("layer.zig").Softmax;
const assert = std.debug.assert;
const std = @import("std");
const t = std.testing;

pub fn Network(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        layers: std.ArrayList(Layer(T)),

        inputs: usize,
        outputs: usize,

        /// Initialize network.
        pub fn init(allocator: Allocator, inputs: usize, outputs: usize) Self {
            return Self{
                .allocator = allocator,
                .layers = std.ArrayList(Layer(T)).init(allocator),
                .inputs = inputs,
                .outputs = outputs,
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self) void {
            for (self.layers.items) |*layer| {
                layer.deinit();
            }
            self.layers.deinit();
        }

        /// Add a layer to the network.
        pub fn addLayer(self: *Self, layer: Layer(T)) !void {
            try self.layers.append(layer);
        }

        /// Return number of nodes in the last layer. If the network does not
        /// have any layer the number of inputs is returned.
        fn numNeuronsOfLastLayer(self: Self) usize {
            return if (self.layers.items.len > 0) self.layers.getLast().getNumOutputs() else self.inputs;
        }

        /// Add sigmoid layer to the network.
        pub fn addSigmoid(self: *Self) !void {
            const dim = self.numNeuronsOfLastLayer();
            const sigmoid = try Sigmoid(T).init(self.allocator, dim);
            const layer = Layer(T){ .sigmoid = sigmoid };
            try self.layers.append(layer);
        }

        /// Add ReLU layer to the network.
        pub fn addReLU(self: *Self) !void {
            const dim = self.numNeuronsOfLastLayer();
            const relu = try ReLU(T).init(self.allocator, dim);
            const layer = Layer(T){ .relu = relu };
            try self.layers.append(layer);
        }

        /// Add softmax layer to the network.
        pub fn addSoftmax(self: *Self) !void {
            const dim = self.numNeuronsOfLastLayer();
            const softmax = try Softmax(T).init(self.allocator, dim);
            const layer = Layer(T){ .softmax = softmax };
            try self.layers.append(layer);
        }

        /// Add linear layer to the network.
        pub fn addLinear(self: *Self, outputs: usize) !void {
            const inputs = self.numNeuronsOfLastLayer();
            const linear = try Linear(T).rand(self.allocator, inputs, outputs);
            const layer = Layer(T){ .linear = linear };
            try self.layers.append(layer);
        }

        /// Feed input through all layers.
        pub fn predict(self: Self, input: Matrix(T)) Matrix(T) {
            var state = input;
            for (self.layers.items) |*layer| {
                state = layer.forward(state);
            }
            return state;
        }

        /// Propagate gradient through layers and adjust parameters.
        pub fn backward(self: Self, input: Matrix(T), labels: Matrix(T), grad: Matrix(T), learning_rate: f32) void {
            var i = self.layers.items.len - 1;
            var err_grad = grad;

            while (true) : (i -= 1) {
                var layer = self.layers.items[i];
                if (i > 0) {
                    const previousLayer = self.layers.items[i - 1];
                    err_grad = layer.backward(previousLayer.getActivation(), labels, err_grad);
                } else {
                    // TODO Don't compute the input gradient for the first layer.
                    err_grad = layer.backward(input, labels, err_grad);
                }

                if (layer == .linear) {
                    layer.linear.applyGradients(learning_rate);
                }

                if (i == 0) break;
            }
        }

        /// Train the network for fixed number of epochs.
        pub fn train(self: Self, epochs: usize, learning_rate: f32, input: Matrix(T), labels: Matrix(T)) !void {
            assert(input.rows == labels.rows);
            assert(input.columns == self.inputs);
            assert(labels.columns == self.outputs);

            // TODO Make loss function configurable
            var cost_fn = try MeanSquaredError(f32).init(self.allocator, self.outputs);
            defer cost_fn.deinit();

            const num_samples = input.rows;
            for (0..epochs) |e| {
                std.debug.print("Epoch {any}\n", .{e});

                for (0..num_samples) |r| {
                    const X = input.getRow(r);
                    const y = labels.getRow(r);
                    const prediction = self.predict(X);
                    const loss = cost_fn.computeLoss(prediction, y);
                    std.debug.print("Forward pass -> Sample: {any} Loss {any}\n", .{ r, loss });

                    const err_grad = cost_fn.computeGradient(prediction, y);
                    self.backward(X, y, err_grad, learning_rate);
                    std.debug.print("Backward pass -> Sample: {any}\n", .{r});
                }
            }
        }
    };
}

test "Make prediction given inputs" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var net = Network(f32).init(allocator, 2, 4);
    defer net.deinit();

    const l1_weights = [_]f32{
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    };
    const l1_biases = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const l1 = try Linear(f32).init(allocator, 2, 4, &l1_weights, &l1_biases);

    const s1 = try Sigmoid(f32).init(allocator, 4);

    try net.addLayer(Layer(f32){ .linear = l1 });
    try net.addLayer(Layer(f32){ .sigmoid = s1 });

    var input_data = [_]f32{ 1.0, 1.0 };
    const input = Matrix(f32).init(1, 2, &input_data);

    const prediction = net.predict(input);
    try t.expectEqualSlices(f32, prediction.elements, &.{ 9.5257413e-1, 9.5257413e-1, 9.5257413e-1, 9.5257413e-1 });
}

test "Train network" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var net = Network(f32).init(allocator, 2, 4);
    defer net.deinit();

    const l1_weights = [_]f32{
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    };
    const l1_biases = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const l1 = try Linear(f32).init(allocator, 2, 4, &l1_weights, &l1_biases);

    const s1 = try Sigmoid(f32).init(allocator, 4);

    try net.addLayer(Layer(f32){ .linear = l1 });
    try net.addLayer(Layer(f32){ .sigmoid = s1 });

    var input_data = [_]f32{ 1.0, 1.0 };
    const input = Matrix(f32).init(1, 2, &input_data);

    var labels_data = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const labels = Matrix(f32).init(1, 4, &labels_data);

    try net.train(40, 0.001, input, labels);
}
