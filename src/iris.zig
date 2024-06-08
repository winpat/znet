const std = @import("std");
const Allocator = std.mem.Allocator;
const CsvReader = @import("csv.zig").CsvReader;
const Matrix = @import("matrix.zig").Matrix;
const eql = std.mem.eql;
const t = std.testing;

const NUM_IRIS_SAMPLES: usize = 150;

// Return iris dataset as a tuple of feature and label matrix.
pub fn load(allocator: Allocator, path: []const u8) !struct { Matrix(f32), Matrix(f32) } {
    var csv = try CsvReader(',').init(allocator, path);
    defer csv.deinit();

    // Skip CSV header
    csv.skipLine();

    var features = try Matrix(f32).alloc(allocator, 150, 4, .zeros);
    var labels = try Matrix(f32).alloc(allocator, 150, 3, .zeros);

    var line: usize = 0;
    while (line < NUM_IRIS_SAMPLES) : (line += 1) {
        const sepal_length = try csv.nextAs(f32);
        features.set(line, 0, sepal_length);

        const sepal_width = try csv.nextAs(f32);
        features.set(line, 1, sepal_width);

        const petal_length = try csv.nextAs(f32);
        features.set(line, 2, petal_length);

        const petal_width = try csv.nextAs(f32);
        features.set(line, 3, petal_width);

        // One hot encode label
        const species = try csv.next();
        if (eql(u8, species, "Setosa")) {
            labels.set(line, 0, 1.0);
        } else if (eql(u8, species, "Versicolor")) {
            labels.set(line, 1, 1.0);
        } else if (eql(u8, species, "Virginica")) {
            labels.set(line, 2, 1.0);
        }
    }

    return .{ features, labels };
}

test "Load iris dataset" {
    const path = "data/iris.csv";
    var features, var labels = try load(t.allocator, path);
    defer {
        features.free(t.allocator);
        labels.free(t.allocator);
    }
    try t.expectEqualSlices(f32, &.{ 5.1, 3.5, 1.4, 0.2 }, features.getRow(0).elements);
    try t.expectEqualSlices(f32, &.{ 1.0, 0.0, 0.0 }, labels.getRow(0).elements);
}
