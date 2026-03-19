const std = @import("std");
const tst = std.testing;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const mem = std.mem;
const RndGen = std.Random.DefaultPrng;

const seed: usize = 0;
var rng = RndGen.init(seed);
const random = rng.random();

pub fn Tensor(T: type, R: usize) type {
    comptime {
        switch (@typeInfo(T)) {
            .float, .int => {},
            else => @compileError("Tensor element type must be numeric"),
        }
    }

    return struct {
        const Self = @This();

        shape: [R]usize,
        strides: [R]usize,
        elements: []T,

        pub fn init(allocator: Allocator, shape: [R]usize) Allocator.Error!Self {
            return .{
                .shape = shape,
                .strides = computeStrides(shape),
                .elements = try allocator.alloc(T, numElements(shape)),
            };
        }

        pub fn zeros(allocator: Allocator, shape: [R]usize) Allocator.Error!Self {
            const tensor = try Self.init(allocator, shape);
            @memset(tensor.elements, 0);
            return tensor;
        }

        pub fn rand(allocator: Allocator, shape: [R]usize) Allocator.Error!Self {
            const tensor = try Self.init(allocator, shape);
            for (tensor.elements) |*v| v.* = random.float(T);
            return tensor;
        }

        pub fn fromSlice(allocator: Allocator, shape: [R]usize, vals: []const T) Allocator.Error!Self {
            const tensor = try Self.init(allocator, shape);
            assert(vals.len == numElements(tensor.shape));
            @memcpy(tensor.elements, vals);
            return tensor;
        }

        pub fn fromBuffer(shape: [R]usize, buf: []T) Self {
            const element_count = numElements(shape);
            assert(buf.len == element_count);
            return .{
                .shape = shape,
                .strides = computeStrides(shape),
                .elements = buf,
            };
        }

        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.free(self.elements);
        }

        pub fn get(self: Self, indices: [R]usize) T {
            const idx = flatIndex(&indices, &self.strides);
            assert(idx < self.elements.len);
            return self.elements[idx];
        }

        pub fn set(self: Self, indices: [R]usize, val: T) void {
            const idx = flatIndex(&indices, &self.strides);
            assert(idx < self.elements.len);
            self.elements[idx] = val;
        }

        pub inline fn rank(self: Self) usize {
            return self.shape.len;
        }
    };
}

fn computeStrides(shape: anytype) @TypeOf(shape) {
    const R = shape.len;
    var strides: [R]usize = undefined;
    strides[R - 1] = 1;
    var i = R - 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

fn numElements(shape: anytype) usize {
    var n: usize = 1;
    for (shape) |d| {
        n *= d;
    }
    return n;
}

fn flatIndex(indices: []const usize, strides: []const usize) usize {
    var idx: usize = 0;
    for (indices, strides) |i, s| idx += i * s;
    return idx;
}

test "Tensor.init()" {
    const tensor = try Tensor(f32, 2).init(tst.allocator, .{ 2, 2 });
    defer tensor.deinit(tst.allocator);

    try tst.expectEqual(.{ 2, 2 }, tensor.shape);
    try tst.expectEqual(.{ 2, 1 }, tensor.strides);
    try tst.expectEqual(4, tensor.elements.len);
}

test "Tensor.zeros()" {
    const tensor = try Tensor(f32, 2).zeros(tst.allocator, .{ 2, 2 });
    defer tensor.deinit(tst.allocator);

    try tst.expectEqual(.{ 2, 2 }, tensor.shape);
    try tst.expectEqual(.{ 2, 1 }, tensor.strides);
    try tst.expectEqual(4, tensor.elements.len);
    try tst.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, tensor.elements);
}

test "Tensor.rand()" {
    const tensor = try Tensor(f32, 2).rand(tst.allocator, .{ 2, 2 });
    defer tensor.deinit(tst.allocator);

    try tst.expectEqual(.{ 2, 2 }, tensor.shape);
    try tst.expectEqual(.{ 2, 1 }, tensor.strides);

    try tst.expectEqual(4, tensor.elements.len);

    var min: f32 = std.math.floatMax(f32);
    var max: f32 = std.math.floatMin(f32);
    for (tensor.elements) |e| {
        min = @min(min, e);
        max = @max(max, e);
    }
    try tst.expect(min != max);
    try tst.expect(min >= 0.0);
    try tst.expect(max <= 1.0);
}

test "Tensor.fromSlice()" {
    const values = &.{ 1, 2, 3, 4 };
    const tensor = try Tensor(f32, 2).fromSlice(tst.allocator, .{ 2, 2 }, values);
    defer tensor.deinit(tst.allocator);

    try tst.expectEqual(.{ 2, 2 }, tensor.shape);
    try tst.expectEqual(.{ 2, 1 }, tensor.strides);

    try tst.expectEqual(4, tensor.elements.len);
    try tst.expectEqualSlices(f32, values, tensor.elements);
}

test "Tensor.fromBuffer()" {
    var buf = [_]f32{ 1, 2, 3, 4 };
    const tensor = Tensor(f32, 2).fromBuffer(.{ 2, 2 }, &buf);

    try tst.expectEqual(.{ 2, 2 }, tensor.shape);
    try tst.expectEqual(.{ 2, 1 }, tensor.strides);

    try tst.expectEqual(4, tensor.elements.len);
    try tst.expectEqualSlices(f32, &buf, tensor.elements);
}

test "Tensor get() and set()" {
    const values: []const f32 = &.{ 1, 2, 3, 4 };
    const tensor = try Tensor(f32, 2).fromSlice(tst.allocator, .{ 2, 2 }, values);
    defer tensor.deinit(tst.allocator);

    try tst.expectEqual(3, tensor.get(.{ 1, 0 }));

    tensor.set(.{ 1, 0 }, 5);
    try tst.expectEqual(5, tensor.get(.{ 1, 0 }));
}

pub fn add(comptime T: type, comptime R: usize, lhs: Tensor(T, R), rhs: Tensor(T, R), res: *Tensor(T, R)) void {
    assert(mem.eql(usize, &lhs.shape, &rhs.shape));
    assert(mem.eql(usize, &lhs.shape, &res.shape));

    for (lhs.elements, rhs.elements, res.elements) |a, b, *r| {
        r.* = a + b;
    }
}

test "add()" {
    const a = try Tensor(f32, 2).fromSlice(tst.allocator, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit(tst.allocator);

    const b = try Tensor(f32, 2).fromSlice(tst.allocator, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit(tst.allocator);

    var r = try Tensor(f32, 2).init(tst.allocator, .{ 2, 2 });
    defer r.deinit(tst.allocator);

    add(f32, 2, a, b, &r);

    try tst.expectEqualSlices(f32, &.{ 6, 8, 10, 12 }, r.elements);
}
