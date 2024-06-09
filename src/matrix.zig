const std = @import("std");
const Allocator = std.mem.Allocator;
const RndGen = std.rand.DefaultPrng;
const assert = std.debug.assert;
const t = std.testing;

const InitStrategy = enum {
    zeros,
    rand,
};

pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        rows: usize,
        columns: usize,
        elements: []T = undefined,

        /// Initalize matrix on unowned slice. The caller is responsible of
        /// ensuring that passed slices lifetime matches the one of the returned
        /// matrix.
        pub fn init(rows: usize, columns: usize, elements: []T) Self {
            assert(rows * columns == elements.len);
            return Self{ .rows = rows, .columns = columns, .elements = elements };
        }

        /// Allocate and initalize matrix. Matrix needs to be freed by the owner.
        pub fn alloc(allocator: Allocator, rows: usize, columns: usize, init_strategy: InitStrategy) !Self {
            const elements = try allocator.alloc(T, rows * columns);
            var matrix = Self.init(rows, columns, elements);
            switch (init_strategy) {
                .zeros => matrix.zeros(),
                .rand => matrix.rand(),
            }
            return matrix;
        }

        /// Allocate a matrix and initalize it with a copy data in passed slice.
        /// The length of the slice needs to match the number of elements.
        pub fn allocFromSlice(allocator: Allocator, rows: usize, columns: usize, data: []const T) !Self {
            const elements = try allocator.alloc(T, rows * columns);
            @memcpy(elements, data);
            return Self.init(rows, columns, elements);
        }

        /// Transpose elements and store them in a newly allocated matrix.
        pub fn allocTranspose(self: Self, allocator: Allocator) !Matrix(T) {
            var transpose = Self.init(
                self.columns,
                self.rows,
                try allocator.alloc(T, self.rows * self.columns),
            );

            for (0..self.rows) |r| {
                for (0..self.columns) |c| {
                    const e = self.get(r, c);
                    transpose.set(c, r, e);
                }
            }

            return transpose;
        }

        /// Release all allocated memory.
        pub fn free(self: Self, allocator: Allocator) void {
            allocator.free(self.elements);
        }

        /// Set all elements to zero.
        pub fn zeros(self: Self) void {
            @memset(self.elements, 0);
        }

        /// Randomly initalize matrix.
        pub fn rand(self: *Self) void {
            var rng = RndGen.init(0);
            const random = rng.random();
            for (self.elements) |*v| {
                v.* = random.float(T);
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
            try writer.print("{}_{}x{}", .{
                T,
                self.rows,
                self.columns,
            });
        }

        /// Invert the sign of every element.
        pub fn negative(self: *Self) void {
            for (self.elements) |*e| {
                e.* *= -1;
            }
        }

        /// Return element at specific row and column.
        pub fn get(self: Self, r: usize, c: usize) T {
            assert(r >= 0 and r < self.rows);
            assert(c >= 0 and c < self.columns);
            return self.elements[r * self.columns + c];
        }

        /// Return row matrix of a given row. The caller is responsible that the
        /// row matrix does not outlive the matrix.
        pub fn getRow(self: Self, r: usize) Self {
            assert(r >= 0 and r < self.rows);
            const from = self.columns * r;
            const to = self.columns * r + self.columns;
            return Self.init(1, self.columns, self.elements[from..to]);
        }

        /// Set element at row and column to value.
        pub fn set(self: *Self, r: usize, c: usize, v: T) void {
            assert(r >= 0 and r < self.rows);
            assert(c >= 0 and c < self.columns);
            self.elements[r * self.columns + c] = v;
        }

        /// Set row to values of a row matrix.
        pub fn setRow(self: *Self, r: usize, m: Self) void {
            assert(r >= 0 and r < self.rows);
            assert(m.rows == 1);
            assert(m.columns == self.columns);
            const dest = self.elements[self.columns * r .. self.columns * r + self.columns];
            @memcpy(dest, m.elements);
        }

        /// Split the matrix into two matrices at request row.
        pub fn splitOnRow(self: *Self, r: usize) struct { Self, Self } {
            assert(r >= 0 and r < self.rows);
            const divider = r * self.columns;
            return .{
                Self.init(r, self.columns, self.elements[0..divider]),
                Self.init(self.rows - r, self.columns, self.elements[divider..self.elements.len]),
            };
        }

        /// Return a row iterator.
        pub fn iterRows(self: Self) RowIterator(T) {
            return RowIterator(T){ .m = self };
        }

        /// Swap two rows with each other.
        pub fn swapRows(self: *Self, a: usize, b: usize) void {
            for (
                self.elements[a * self.columns .. a * self.columns + self.columns],
                self.elements[b * self.columns .. b * self.columns + self.columns],
            ) |*x, *y| {
                const tmp = x.*;
                x.* = y.*;
                y.* = tmp;
            }
        }

        /// Randomly shuffle rows.
        pub fn shuffleRows(self: *Self) !void {
            var prng = std.rand.DefaultPrng.init(0);
            const random = prng.random();

            var i = self.rows - 1;
            while (i > 0) : (i -= 1) {
                self.swapRows(i, random.intRangeLessThan(usize, 0, i));
            }
        }

        /// Copy all elements from matrix with the same dimensions.
        pub fn copyFrom(self: *Self, other: Matrix(T)) void {
            assert(other.rows == self.rows and other.columns == self.columns);
            @memcpy(self.elements, other.elements);
        }

        /// Check if the two matrices have same dimensions.
        pub fn sameDimAs(self: Self, other: Matrix(T)) bool {
            return self.rows == other.rows and self.columns == other.columns;
        }

        /// Print the matrix including it's elements. Useful for debugging.
        pub fn print(self: Self) void {
            std.debug.print("{d}x{d} - {any}\n", .{ self.rows, self.columns, self.elements });
        }
    };
}

pub fn RowIterator(comptime T: type) type {
    return struct {
        const Self = @This();

        m: Matrix(T),
        current_row: usize = 0,

        pub fn next(self: *Self) ?Matrix(T) {
            if (self.current_row == self.m.rows) {
                return null;
            }

            const from = self.current_row * self.m.columns;
            const to = (self.current_row * self.m.columns) + self.m.columns;

            defer self.current_row += 1;
            return Matrix(T).init(1, self.m.columns, self.m.elements[from..to]);
        }
    };
}

test "Set and get value" {
    var m_e = [_]f32{0.0} ** 2;
    var m = Matrix(f32).init(1, 2, &m_e);

    try t.expectEqual(m.get(0, 0), 0);
    m.set(0, 0, 1);
    try t.expectEqual(m.get(0, 0), 1);
}

test "Set and get row" {
    var m_e = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var m = Matrix(f32).init(3, 2, &m_e);

    try t.expectEqualSlices(f32, m.getRow(0).elements, &.{ 1.0, 2.0 });
    try t.expectEqualSlices(f32, m.getRow(1).elements, &.{ 3.0, 4.0 });
    try t.expectEqualSlices(f32, m.getRow(2).elements, &.{ 5.0, 6.0 });

    var new_row = [_]f32{ 7.0, 8.0 };
    const row_matrix = Matrix(f32).init(1, 2, &new_row);

    m.setRow(2, row_matrix);
    try t.expectEqualSlices(f32, m.getRow(2).elements, row_matrix.elements);
}

test "Iterate over matrix rows" {
    var m_e = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var m = Matrix(f32).init(2, 2, &m_e);

    var row_iterator = m.iterRows();
    try t.expectEqualSlices(f32, row_iterator.next().?.elements, &.{ 1.0, 2.0 });
    try t.expectEqualSlices(f32, row_iterator.next().?.elements, &.{ 3.0, 4.0 });
    try t.expectEqual(row_iterator.next(), null);
}

test "Swap two rows" {
    var m_e = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var m = Matrix(f32).init(2, 2, &m_e);

    m.swapRows(0, 1);
    try t.expectEqualSlices(f32, m.elements, &.{ 3.0, 4.0, 1.0, 2.0 });
}

test "Shuffle matrix" {
    var m_e = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var m = Matrix(f32).init(2, 2, &m_e);

    try m.shuffleRows();
    try t.expectEqualSlices(f32, &.{ 3.0, 4.0, 1.0, 2.0 }, m.elements);
}

test "Split matrix on row" {
    var m_e = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var m = Matrix(f32).init(5, 1, &m_e);

    const low, const high = m.splitOnRow(2);

    try t.expectEqualSlices(f32, low.elements, &.{ 1.0, 2.0 });
    try t.expectEqual(low.rows, 2);

    try t.expectEqualSlices(f32, high.elements, &.{ 3.0, 4.0, 5.0 });
    try t.expectEqual(high.rows, 3);
}

test "Allocate matrix and initalize with zeros" {
    var m = try Matrix(f32).alloc(t.allocator, 1, 2, .zeros);
    defer m.free(t.allocator);

    try t.expectEqual(m.rows, 1);
    try t.expectEqual(m.columns, 2);
    try t.expectEqualSlices(f32, m.elements, &.{ 0.0, 0.0 });
}

test "Allocate matrix and initalize with random numbers" {
    var m = try Matrix(f32).alloc(t.allocator, 1, 2, .rand);
    defer m.free(t.allocator);

    try t.expectEqual(m.rows, 1);
    try t.expectEqual(m.columns, 2);
    for (m.elements) |e| {
        try t.expect(e >= 0 and e < 1);
    }
}

test "Allocate matrix and initalize it from slice" {
    const data = [_]f32{ 2.0, 3.0 };

    var m = try Matrix(f32).allocFromSlice(t.allocator, 1, 2, &data);
    defer m.free(t.allocator);

    try t.expectEqual(m.rows, 1);
    try t.expectEqual(m.columns, 2);
    try t.expectEqualSlices(f32, m.elements, &data);
}

test "Transpose matrix elements" {
    const data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };

    var m = try Matrix(f32).allocFromSlice(t.allocator, 2, 3, &data);
    defer m.free(t.allocator);

    const transpose = try m.allocTranspose(t.allocator);
    defer transpose.free(t.allocator);

    try t.expectEqualSlices(f32, transpose.elements, &.{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    });
}

test "Copy elements from other matrix" {
    var a = try Matrix(f32).alloc(t.allocator, 1, 2, .zeros);
    defer a.free(t.allocator);

    var b = try Matrix(f32).alloc(t.allocator, 1, 2, .rand);
    defer b.free(t.allocator);

    a.copyFrom(b);
    try t.expectEqualSlices(f32, a.elements, b.elements);
}

test "Invert sign of matrix elements" {
    var data = [_]f32{ 1.0, -2.0, 3.0 };
    var m = Matrix(f32).init(1, 3, &data);

    m.negative();
    try t.expectEqualSlices(f32, m.elements, &.{ -1.0, 2.0, -3.0 });
}

test "Check if Matrix has same dimension as other" {
    var a = try Matrix(f32).alloc(t.allocator, 1, 2, .zeros);
    defer a.free(t.allocator);

    var b = try Matrix(f32).alloc(t.allocator, 2, 2, .zeros);
    defer b.free(t.allocator);

    try t.expect(a.sameDimAs(a));
    try t.expect(!a.sameDimAs(b));
}
