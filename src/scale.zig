const std = @import("std");
const tst = std.testing;

const Matrix = @import("matrix.zig").Matrix;

/// Normalize matrix columns by it's min and max values.
pub fn minMaxNormalize(comptime T: type, mat: *Matrix(T)) void {
    const rows, const cols = mat.shape;

    for (0..cols) |c| {
        var min: T = mat.get(.{ 0, c });
        var max: T = min;
        for (1..rows) |r| {
            const v = mat.get(.{ r, c });
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const diff = max - min;
        for (0..rows) |r| {
            const v = mat.get(.{ r, c });
            const v_norm = (v - min) / diff;
            mat.set(.{ r, c }, v_norm);
        }
    }
}

test "Test min-max normalization of matrix columns" {
    var data = [_]f32{
        3.0, 6.0, 17.0,
        8.0, 1.0, 7.0,
        9.0, 7.0, 5.0,
    };
    var mat = Matrix(f32).fromBuffer(.{ 3, 3 }, &data);

    minMaxNormalize(f32, &mat);

    try tst.expectEqualSlices(
        f32,
        &.{
            0e0,         8.333333e-1, 1e0,
            8.333333e-1, 0e0,         1.6666667e-1,
            1e0,         1e0,         0e0,
        },
        mat.elements,
    );
}
