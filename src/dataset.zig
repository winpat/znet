const std = @import("std");
const Allocator = std.mem.Allocator;
const eql = std.mem.eql;
const tst = std.testing;

const csv = @import("csv.zig");
const mat = @import("matrix.zig");
const Matrix = mat.Matrix;

const iris_samples_num: usize = 150;

// Return iris dataset as a tuple of feature and label matrix.
pub fn load(allocator: Allocator, path: []const u8) !struct { Matrix(f32), Matrix(f32) } {
    var reader = try csv.Reader(',').init(allocator, path);
    defer reader.deinit();

    // Skip CSV header
    reader.skipLine();

    var features = try Matrix(f32).zeros(allocator, .{ 150, 4 });
    var labels = try Matrix(f32).zeros(allocator, .{ 150, 3 });

    var line: usize = 0;
    while (line < iris_samples_num) : (line += 1) {
        const sepal_length = try reader.nextAs(f32);
        features.set(.{ line, 0 }, sepal_length);

        const sepal_width = try reader.nextAs(f32);
        features.set(.{ line, 1 }, sepal_width);

        const petal_length = try reader.nextAs(f32);
        features.set(.{ line, 2 }, petal_length);

        const petal_width = try reader.nextAs(f32);
        features.set(.{ line, 3 }, petal_width);

        // One hot encode label
        const species = try reader.next();
        if (eql(u8, species, "Setosa")) {
            labels.set(.{ line, 0 }, 1.0);
        } else if (eql(u8, species, "Versicolor")) {
            labels.set(.{ line, 1 }, 1.0);
        } else if (eql(u8, species, "Virginica")) {
            labels.set(.{ line, 2 }, 1.0);
        }
    }

    return .{ features, labels };
}

test "Load iris dataset" {
    const path = "data/iris.csv";
    var features, var labels = try load(tst.allocator, path);
    defer {
        features.deinit(tst.allocator);
        labels.deinit(tst.allocator);
    }
    const first_features = mat.row(f32, &features, 0);
    try tst.expectEqualSlices(f32, &.{ 5.1, 3.5, 1.4, 0.2 }, first_features.elements);
    const first_labels = mat.row(f32, &labels, 0);
    try tst.expectEqualSlices(f32, &.{ 1.0, 0.0, 0.0 }, first_labels.elements);
}
