const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Writer = std.Io.Writer;
const StringHashMap = std.StringArrayHashMapUnmanaged;
const mem = std.mem;

pub const App = struct {
    name: []const u8,
    version: ?[]const u8 = null,
    args: []const Argument,
    opts: []const Option,

    pub fn parse(self: App, allocator: Allocator) !struct { StringHashMap(Value), StringHashMap(Value) } {
        var args = StringHashMap(Value){};
        var opts = StringHashMap(Value){};

        var arg_pos: usize = 0;

        for (self.opts) |opt| {
            if (opt.default) |default|
                try opts.put(allocator, opt.name, default);
        }

        for (self.args) |arg| {
            if (arg.default) |default|
                try args.put(allocator, arg.name, default);
        }

        var iter = std.process.ArgIterator.init();
        _ = iter.skip();

        while (iter.next()) |tk| {
            if (tk[0] == '-') {
                const is_long = tk.len > 1 and tk[1] == '-';
                const name = if (is_long) tk[2..] else tk[1..];

                if (name.len == 0)
                    return error.MissingOptionName;

                const opt = self.findOption(name) orelse
                    return error.UnknownOption;

                if (opt.flag) {
                    try opts.put(allocator, name, .{ .boolean = true });
                } else {
                    const val = iter.next() orelse
                        return error.OptionMissingValue;

                    try opts.put(allocator, opt.name, try Value.parse(val, opt.kind));
                }
            } else {
                if (arg_pos >= self.args.len)
                    return error.UnknownArgument;

                const arg = self.args[arg_pos];
                defer arg_pos += 1;

                try args.put(allocator, arg.name, try Value.parse(tk, arg.kind));
            }
        }

        return .{ opts, args };
    }

    fn findOption(self: App, name: []const u8) ?Option {
        for (self.opts) |opt| {
            if (mem.eql(u8, opt.name, name) or
                opt.long != null and mem.eql(u8, opt.long.?, name) or
                opt.short != null and mem.eql(u8, opt.short.?, name))
            {
                return opt;
            }
        }
        return null;
    }
};

const Argument = struct {
    name: []const u8,
    kind: ValueKind = .string,
    default: ?Value = null,
};

const Option = struct {
    name: []const u8,
    long: ?[]const u8 = null,
    short: ?[]const u8 = null,
    flag: bool = false,
    kind: ValueKind = .string,
    default: ?Value = null,
};

const ValueKind = enum {
    string,
    float,
    int,
    boolean,
};

const Value = union(ValueKind) {
    string: []const u8,
    float: f64,
    int: i64,
    boolean: bool,

    pub fn format(self: Value, writer: *Writer) Writer.Error!void {
        try switch (self) {
            .string => |string| writer.print("{s}", .{string}),
            .float => |float| writer.print("{d}", .{float}),
            .int => |int| writer.print("{d}", .{int}),
            .boolean => |boolean| writer.print("{}", .{boolean}),
        };
    }

    fn parse(val: []const u8, kind: ValueKind) !Value {
        return switch (kind) {
            .string => .{ .string = val },
            .float => .{ .float = try std.fmt.parseFloat(f64, val) },
            .int => .{ .int = try std.fmt.parseInt(i64, val, 10) },
            .boolean => blk: {
                if (mem.eql(u8, val, "true"))
                    break :blk .{ .boolean = true };

                if (mem.eql(u8, val, "false"))
                    break :blk .{ .boolean = false };

                return error.BooleanParseError;
            },
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var app = App{
        .name = "iris",
        .args = &.{
            .{ .name = "epochs", .default = .{ .int = 30 } },
        },
        .opts = &.{
            .{ .name = "verbose", .flag = true, .default = .{ .boolean = false } },
        },
    };

    const opts, const args = try app.parse(allocator);

    {
        var iter = args.iterator();
        while (iter.next()) |entry| {
            std.debug.print(
                "{s}: {f}\n",
                .{ entry.key_ptr.*, entry.value_ptr.* },
            );
        }
    }

    {
        var iter = opts.iterator();
        while (iter.next()) |entry| {
            std.debug.print(
                "{s}: {f}\n",
                .{ entry.key_ptr.*, entry.value_ptr.* },
            );
        }
    }
}
