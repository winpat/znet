const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const t = std.testing;

/// Normalize matrix columns by it's min and max values.
pub fn minMaxNormalize(comptime T: type, m: *Matrix(T)) void {
    for (0..m.columns) |c| {
        var min: T = m.get(0, c);
        var max: T = min;
        for (1..m.rows) |r| {
            const e = m.get(r, c);
            if (e < min) min = e;
            if (e > max) max = e;
        }

        const diff = max - min;
        for (0..m.rows) |r| {
            const e = m.get(r, c);
            const e_norm = (e - min) / diff;
            m.set(r, c, e_norm);
        }
    }
}

test "Test min-max normalization of matrix columns" {
    var data = [_]f32{
        3.0, 6.0, 17.0,
        8.0, 1.0, 7.0,
        9.0, 7.0, 5.0,
    };
    var m = Matrix(f32).init(3, 3, &data);

    minMaxNormalize(f32, &m);
    try t.expectEqualSlices(f32, &.{
        0e0,         8.333333e-1, 1e0,
        8.333333e-1, 0e0,         1.6666667e-1,
        1e0,         1e0,         0e0,
    }, m.elements);
}
