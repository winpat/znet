const std = @import("std");
const Allocator = std.mem.Allocator;
const fs = std.fs;
const fmt = std.fmt;
const tst = std.testing;

pub const ReadError = error{
    EndOfFile,
    ParseError,
};

pub fn Reader(comptime D: comptime_int) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        buffer: []u8,
        pos: usize = 0,

        /// Initialize CSV reader.
        pub fn init(allocator: Allocator, path: []const u8) !Self {
            var file = try fs.cwd().openFile(path, .{ .mode = .read_only });
            const fstat = try file.stat();

            const buffer = try allocator.alloc(u8, fstat.size);
            errdefer allocator.free(buffer);

            _ = try file.read(buffer);

            return Self{
                .allocator = allocator,
                .buffer = buffer,
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: Self) void {
            self.allocator.free(self.buffer);
        }

        /// Skip line.
        pub fn skipLine(self: *Self) void {
            while (self.pos < self.buffer.len) : (self.pos += 1) {
                if (self.buffer[self.pos] == '\n') {
                    self.pos += 1;
                    break;
                }
            }
        }

        /// Read next column.
        pub fn next(self: *Self) ![]const u8 {
            const start = self.pos;
            if (start == self.buffer.len) return ReadError.EndOfFile;

            return while (self.pos < self.buffer.len) : (self.pos += 1) {
                switch (self.buffer[self.pos]) {
                    D, '\n' => {
                        defer self.pos += 1;
                        break self.buffer[start..self.pos];
                    },
                    else => continue,
                }
            } else self.buffer[start..self.pos];
        }

        /// Read next column and try to parse it.
        pub fn nextAs(self: *Self, comptime T: type) !T {
            const v = try self.next();
            return switch (@typeInfo(T)) {
                .float => fmt.parseFloat(T, v) catch return ReadError.ParseError,
                else => @compileError("Datatype is not implemented."),
            };
        }
    };
}

test "Iterate over CSV" {
    var csv = try Reader(',').init(tst.allocator, "data/example.csv");
    defer csv.deinit();

    csv.skipLine();
    try tst.expectEqual(csv.pos, 4);

    try tst.expectEqualSlices(u8, "1.0", try csv.next());
    try tst.expectEqualSlices(u8, "2.0", try csv.next());
    try tst.expectEqual(@as(f32, 3.0), try csv.nextAs(f32));
    try tst.expectEqual(@as(f64, 4.0), try csv.nextAs(f64));
    try tst.expectError(ReadError.EndOfFile, csv.next());
}
