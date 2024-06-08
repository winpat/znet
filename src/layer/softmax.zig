const std = @import("std");
const Allocator = std.mem.Allocator;
const Matrix = @import("../matrix.zig").Matrix;
const ops = @import("../ops.zig");
const t = std.testing;
const assert = std.debug.assert;

pub fn Softmax(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        dim: usize,

        activations: Matrix(T),
        gradient: Matrix(T),

        /// Initialize softmax layer.
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
        pub fn backward(self: Self, labels: Matrix(T), err_grad: Matrix(T)) Matrix(T) {
            assert(self.activations.sameDimAs(labels));
            assert(self.activations.sameDimAs(err_grad));

            _, const predicted_class_idx = ops.argmax(T, labels);
            const predicted_prop = labels.get(0, predicted_class_idx);

            for (self.gradient.elements, self.activations.elements, err_grad.elements, 0..) |*g, a, e, c| {
                if (c == predicted_class_idx) {
                    g.* = a * (1 - a) * e;
                } else {
                    g.* = -predicted_prop * a * e;
                }
            }
            return self.gradient;
        }
    };
}

test "Softmax forward pass" {
    var input_data = [_]f32{ 1, 2, 3 };
    const input = Matrix(f32).init(1, 3, &input_data);

    var softmax = try Softmax(f32).init(t.allocator, 3);
    defer softmax.deinit();

    const prediction = softmax.forward(input);

    try t.expectEqualSlices(
        f32,
        &.{ 9.003057e-2, 2.4472848e-1, 6.6524094e-1 },
        prediction.elements,
    );
}

test "Softmax backward pass" {
    var input_data = [_]f32{ 0.2, 0.3, 0.4 };
    const input = Matrix(f32).init(1, 3, &input_data);

    var label_data = [_]f32{ 0.0, 0.0, 1.0 };
    const labels = Matrix(f32).init(1, 3, &label_data);

    var err_grad_data = [_]f32{ 0.5, 0.5, 0.5 };
    const err_grad = Matrix(f32).init(1, 3, &err_grad_data);

    var softmax = try Softmax(f32).init(t.allocator, 3);
    defer softmax.deinit();

    _ = softmax.forward(input);
    const grad = softmax.backward(labels, err_grad);
    try t.expectEqualSlices(
        f32,
        &.{ -1.5030481e-1, -1.661125e-1, 1.16177484e-1 },
        grad.elements,
    );
}
