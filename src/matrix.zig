const std = @import("std");
const tst = std.testing;
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = std.mem.Allocator;

const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

pub fn Matrix(T: type) type {
    return Tensor(T, 2);
}

pub fn add(comptime T: type, lhs: Matrix(T), rhs: Matrix(T), res: *Matrix(T)) void {
    tensor.add(T, 2, lhs, rhs, res);
}

test "add()" {
    const a = try Matrix(f32).fromSlice(tst.allocator, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit(tst.allocator);

    const b = try Matrix(f32).fromSlice(tst.allocator, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit(tst.allocator);

    var r = try Matrix(f32).init(tst.allocator, .{ 2, 2 });
    defer r.deinit(tst.allocator);

    add(f32, a, b, &r);

    try tst.expectEqualSlices(f32, &.{ 6, 8, 10, 12 }, r.elements);
}

/// Compute the dot product between two matrices.
///
/// The number of columns of the first matrix needs to
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
pub fn mul(comptime T: type, lhs: Matrix(T), rhs: Matrix(T), res: *Matrix(T)) void {
    const M = lhs.shape[0];
    const K = lhs.shape[1];
    assert(rhs.shape[0] == K);

    const N = rhs.shape[1];
    assert(res.shape[0] == M);
    assert(res.shape[1] == N);

    for (0..M) |i| {
        for (0..N) |j| {
            var val: T = 0;
            for (0..K) |k| {
                val += lhs.get(.{ i, k }) * rhs.get(.{ k, j });
            }
            res.set(.{ i, j }, val);
        }
    }
}

test "mul()" {
    // [ 1 2 3 ]   [ 7  8  ]   [ 58  64  ]
    // [ 4 5 6 ] * [ 9  10 ] = [ 139 154 ]
    //             [ 11 12 ]
    const a = try Matrix(f32).fromSlice(tst.allocator, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit(tst.allocator);

    const b = try Matrix(f32).fromSlice(tst.allocator, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit(tst.allocator);

    var r = try Matrix(f32).init(tst.allocator, .{ 2, 2 });
    defer r.deinit(tst.allocator);

    mul(f32, a, b, &r);

    try tst.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, r.elements);
}

pub fn row(comptime T: type, mat: *Matrix(T), r: usize) Matrix(T) {
    const rows, const cols = mat.shape;
    assert(r < rows);
    const offset = r * cols;
    return Matrix(T).fromBuffer(.{ 1, cols }, mat.elements[offset .. offset + cols]);
}

test "row()" {
    var mat = try Matrix(f32).fromSlice(tst.allocator, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer mat.deinit(tst.allocator);

    const first_row = row(f32, &mat, 0);
    try tst.expectEqual(.{ 1, 3 }, first_row.shape);
    try tst.expectEqualSlices(f32, &.{ 1, 2, 3 }, first_row.elements[0..3]);

    const second_row = row(f32, &mat, 1);
    try tst.expectEqual(.{ 1, 3 }, second_row.shape);
    try tst.expectEqualSlices(f32, &.{ 4, 5, 6 }, second_row.elements);
}

pub fn swapRows(comptime T: type, mat: *Matrix(T), a: usize, b: usize) void {
    assert(a < mat.shape[0]);
    assert(b < mat.shape[0]);
    const cols = mat.shape[1];
    for (mat.elements[a * cols ..][0..cols], mat.elements[b * cols ..][0..cols]) |*x, *y| {
        const tmp = x.*;
        x.* = y.*;
        y.* = tmp;
    }
}

test "swapRows()" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var mat = Matrix(f32).fromBuffer(.{ 3, 2 }, &data);

    swapRows(f32, &mat, 0, 2);
    try tst.expectEqualSlices(f32, &.{ 5, 6, 3, 4, 1, 2 }, mat.elements);
}

/// Split a matrix into two at the given row. Returns two non-owning views:
/// the first contains rows [0, row), the second contains rows [row, rows).
pub fn split(comptime T: type, mat: *Matrix(T), r: usize) struct { Matrix(T), Matrix(T) } {
    assert(r < mat.shape[0]);
    const cols = mat.shape[1];
    const divider = r * cols;
    return .{
        Matrix(T).fromBuffer(.{ r, cols }, mat.elements[0..divider]),
        Matrix(T).fromBuffer(.{ mat.shape[0] - r, cols }, mat.elements[divider..]),
    };
}

test "split()" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var mat = Matrix(f32).fromBuffer(.{ 3, 2 }, &data);

    const top, const bottom = split(f32, &mat, 1);

    try tst.expectEqual(.{ 1, 2 }, top.shape);
    try tst.expectEqualSlices(f32, &.{ 1, 2 }, top.elements);

    try tst.expectEqual(.{ 2, 2 }, bottom.shape);
    try tst.expectEqualSlices(f32, &.{ 3, 4, 5, 6 }, bottom.elements);
}

/// Transpose a matrix. Writes into a pre-allocated result matrix.
pub fn transpose(comptime T: type, mat: Matrix(T), res: *Matrix(T)) void {
    assert(mat.shape[0] == res.shape[1]);
    assert(mat.shape[1] == res.shape[0]);
    const rows, const cols = mat.shape;
    for (0..rows) |r| {
        for (0..cols) |c| {
            res.set(.{ c, r }, mat.get(.{ r, c }));
        }
    }
}

test "transpose()" {
    const mat = try Matrix(f32).fromSlice(tst.allocator, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer mat.deinit(tst.allocator);

    var res = try Matrix(f32).init(tst.allocator, .{ 3, 2 });
    defer res.deinit(tst.allocator);

    transpose(f32, mat, &res);
    try tst.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, res.elements);
}

/// Set a row of a matrix from a row matrix.
pub fn setRow(comptime T: type, mat: *Matrix(T), r: usize, data: Matrix(T)) void {
    assert(r < mat.shape[0]);
    assert(data.shape[1] == mat.shape[1]);
    const cols = mat.shape[1];
    const start = r * cols;
    @memcpy(mat.elements[start..][0..cols], data.elements[0..cols]);
}

test "setRow()" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var mat = Matrix(f32).fromBuffer(.{ 3, 2 }, &data);

    var row_data = [_]f32{ 9, 8 };
    const r = Matrix(f32).fromBuffer(.{ 1, 2 }, &row_data);

    setRow(f32, &mat, 1, r);
    try tst.expectEqualSlices(f32, &.{ 1, 2, 9, 8, 5, 6 }, mat.elements);
}

pub fn argmax(comptime T: type, mat: Matrix(T)) usize {
    const rows, const cols = mat.shape;
    assert(rows == 1 or cols == 1);

    var val_max: T = mat.elements[0];
    var idx: usize = 0;
    for (mat.elements[1..], 1..) |e, i| {
        if (e > val_max) {
            val_max = e;
            idx = i;
        }
    }
    return idx;
}

test "argmax()" {
    var mat = try Matrix(f32).fromSlice(tst.allocator, .{ 1, 6 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer mat.deinit(tst.allocator);

    try tst.expectEqual(argmax(f32, mat), 5);
}
