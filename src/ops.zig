const std = @import("std");
const assert = std.debug.assert;
const t = std.testing;

const Matrix = @import("matrix.zig").Matrix;

/// Compute the mean of all matrix elements.
pub fn mean(comptime T: type, m: Matrix(T)) T {
    var sum: T = 0;
    for (m.elements) |e| {
        sum += e;
    }
    return sum / @as(T, @floatFromInt(m.elements.len));
}

/// Add two matrices together.
pub fn add(comptime T: type, a: Matrix(T), b: Matrix(T), r: *Matrix(T)) void {
    for (a.elements, b.elements, 0..) |x, y, idx| {
        r.elements[idx] = x + y;
    }
}

/// Add scalar to matrix.
pub fn addScalar(comptime T: type, a: Matrix(T), b: T, r: *Matrix(T)) void {
    for (a.elements, r.elements) |x, *z| {
        z.* = x + b;
    }
}

/// Subtract matrix from another.
pub fn subtract(comptime T: type, a: Matrix(T), b: Matrix(T), r: *Matrix(T)) void {
    for (a.elements, b.elements, 0..) |x, y, idx| {
        r.elements[idx] = x - y;
    }
}

/// Flip the sign of all matrix elements.
pub fn flipSign(comptime T: type, a: Matrix(T), r: *Matrix(T)) void {
    for (a.elements, r.elements) |x, *z| {
        z.* = x * -1;
    }
}

/// Multiply two matrices with each other.
///
/// For this to work the number of columns of the first matrix needs to
/// be equal to the number of rows in the second matrix.
///
/// The resulting matrix will have the number of rows of the first matrix
/// and the number columns of the second matrix.
///
/// MxN * NxP = MxP
///
///                 [ y1 y4 ]
///  [ x1 x2 x3 ] * [ y2 y5 ] = [ x1*y1+x2*y2+x3*y3, x1*y4+x2*y5+x3*y6 ]
///  [ x4 x5 x6 ]   [ y3 y6 ]   [ x4*y1+x5*y2+x6*y3, x4*y4+x5*y5+x6*y6 ]
///
pub fn multiply(
    comptime T: type,
    a: Matrix(T),
    b: Matrix(T),
    r: *Matrix(T),
) void {
    assert(a.columns == b.rows);
    assert(r.rows == a.rows and r.columns == b.columns);

    for (0..a.rows) |i| {
        for (0..b.columns) |j| {
            var v: T = 0;
            for (0..a.columns) |k| {
                v += a.get(i, k) * b.get(k, j);
            }
            r.set(i, j, v);
        }
    }
}

/// Multiply elements of two matrices pair-wise.
pub fn hadmardProduct(
    comptime T: type,
    a: Matrix(T),
    b: Matrix(T),
    r: *Matrix(T),
) void {
    for (a.elements, b.elements, 0..) |x, y, idx| {
        r.elements[idx] = x * y;
    }
}

/// Multiply matrix with a scalar.
pub fn multiplyScalar(
    comptime T: type,
    m: Matrix(T),
    s: T,
    r: *Matrix(T),
) void {
    for (m.elements, 0..) |x, idx| {
        r.elements[idx] = x * s;
    }
}

/// Map a unary operation across all matrix elements.
pub fn map(comptime T: type, m: *Matrix(T), op: *const fn (T) T) void {
    for (m.elements, m.elements) |*e, v| {
        e.* = op(v);
    }
}

/// Return the row and column of the largest element in a matrix.
pub fn argmax(comptime T: type, m: Matrix(T)) usize {
    assert(m.rows == 1 or m.columns == 1);
    var max_value: T = m.elements[0];
    var index: usize = 0;
    for (m.elements[1..], 1..) |e, idx| {
        if (e > max_value) {
            max_value = e;
            index = idx;
        }
    }
    return index;
}

test "Add two matrices together" {
    var a_e = [_]f32{2.0} ** 4;
    const a = Matrix(f32).fromSlice(2, 2, &a_e);

    var b_e = [_]f32{1.0} ** 4;
    const b = Matrix(f32).fromSlice(2, 2, &b_e);

    var r_e = [_]f32{0.0} ** 4;
    var r = Matrix(f32).fromSlice(2, 2, &r_e);

    add(f32, a, b, &r);
    try t.expectEqualSlices(f32, r.elements, &.{ 3.0, 3.0, 3.0, 3.0 });
}

test "Add scalar to matrix" {
    var a_e = [_]f32{2.0} ** 4;
    const a = Matrix(f32).fromSlice(2, 2, &a_e);

    var r_e = [_]f32{0.0} ** 4;
    var r = Matrix(f32).fromSlice(2, 2, &r_e);

    addScalar(f32, a, 1.0, &r);
    try t.expectEqualSlices(f32, r.elements, &.{ 3.0, 3.0, 3.0, 3.0 });
}

test "Subtract one matrix from another" {
    var a_e = [_]f32{3.0} ** 4;
    const a = Matrix(f32).fromSlice(2, 2, &a_e);

    var b_e = [_]f32{1.0} ** 4;
    const b = Matrix(f32).fromSlice(2, 2, &b_e);

    var r_e = [_]f32{0.0} ** 4;
    var r = Matrix(f32).fromSlice(2, 2, &r_e);

    subtract(f32, a, b, &r);
    try t.expectEqualSlices(f32, r.elements, &.{ 2.0, 2.0, 2.0, 2.0 });
}

test "Flip sign of matrix elements" {
    var data = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    var m = Matrix(f32).fromSlice(2, 2, &data);

    flipSign(f32, m, &m);
    try t.expectEqualSlices(f32, m.elements, &.{ -1.0, 2.0, -3.0, 4.0 });
}

test "Multiply two matrices together" {
    var a = [_]f32{ 5.1, 3.5, 1.4, 0.2 };
    const am = Matrix(f32).fromSlice(1, 4, &a);

    var b = [_]f32{
        0.01, 0.01, 0.01, 0.01, 0.02, 0.02,
        0.02, 0.02, 0.03, 0.03, 0.03, 0.03,
        0.04, 0.04, 0.04, 0.04, 0.05, 0.05,
        0.05, 0.05, 0.6,  0.6,  0.6,  0.6,
    };
    const bm = Matrix(f32).fromSlice(4, 6, &b);

    var r = [_]f32{0} ** 6;
    var rm = Matrix(f32).fromSlice(1, 6, &r);

    // 1x4 * 4x6 = 1x6
    multiply(f32, am, bm, &rm);
    try t.expectEqualSlices(f32, rm.elements, &.{ 1.87e-1, 1.87e-1, 3.32e-1, 3.32e-1, 3.9699998e-1, 3.9699998e-1 });
}

test "Multiply two matrices element wise" {
    var a_e = [_]f32{2.0} ** 4;
    const a = Matrix(f32).fromSlice(2, 2, &a_e);

    var b_e = [_]f32{2.0} ** 4;
    const b = Matrix(f32).fromSlice(2, 2, &b_e);

    var r_e = [_]f32{0.0} ** 4;
    var r = Matrix(f32).fromSlice(2, 2, &r_e);

    hadmardProduct(f32, a, b, &r);
    try t.expectEqualSlices(f32, r.elements, &.{ 4.0, 4.0, 4.0, 4.0 });
}

test "Multiply matrix with a scalar element wise" {
    var a_e = [_]f32{2.0} ** 4;
    const a = Matrix(f32).fromSlice(2, 2, &a_e);

    const s = 2.0;

    var r_e = [_]f32{0.0} ** 4;
    var r = Matrix(f32).fromSlice(2, 2, &r_e);

    multiplyScalar(f32, a, s, &r);
    try t.expectEqualSlices(f32, r.elements, &.{ 4.0, 4.0, 4.0, 4.0 });
}

test "Get the mean value of matrix" {
    var m_e = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const m = Matrix(f32).fromSlice(2, 2, &m_e);

    try t.expectEqual(mean(f32, m), 2.5);
}

fn addOne(v: f32) f32 {
    return v + 1;
}

test "Map unary operation over matrix elements" {
    var m_e = [_]f32{2.0} ** 4;
    var m = Matrix(f32).fromSlice(2, 2, &m_e);

    map(f32, &m, addOne);
    try t.expectEqualSlices(f32, m.elements, &.{ 3.0, 3.0, 3.0, 3.0 });
}

test "Get index highest of largest element in matrix" {
    var m_e = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const m = Matrix(f32).fromSlice(1, 4, &m_e);

    try t.expectEqual(argmax(f32, m), 3);
}
