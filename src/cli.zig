const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const process = std.process;
const ArrayList = std.ArrayList;
const Writer = std.Io.Writer;
const StringHashMap = std.StringArrayHashMapUnmanaged;
const mem = std.mem;

pub const App = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    version: ?[]const u8 = null,
    root: Command,
};

const Command = struct {
    name: []const u8 = "",
    description: ?[]const u8 = null,
    cmds: []const Command = &.{},
    args: []const Argument = &.{},
    opts: []const Option = &.{},
};

const Option = struct {
    name: []const u8,
    long: ?[]const u8 = null,
    short: ?[]const u8 = null,
    description: ?[]const u8 = null,
    flag: bool = false,
    kind: ValueKind = .string,
    default: ?Value = null,
};

const Argument = struct {
    name: []const u8,
    description: ?[]const u8 = null,
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
            .boolean => {
                if (mem.eql(u8, val, "true"))
                    return .{ .boolean = true };

                if (mem.eql(u8, val, "false"))
                    return .{ .boolean = false };

                return error.BooleanParseError;
            },
        };
    }
};

var stdout_buf = [_]u8{0} ** 1024;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);

var stderr_buf = [_]u8{0} ** 1024;
var stderr_writer = std.fs.File.stderr().writer(&stderr_buf);

var stdin_buf = [_]u8{0} ** 1024;
var stdin_reader = std.fs.File.stdin().reader(&stdin_buf);

pub const Context = struct {
    allocator: Allocator,

    app: *const App,

    path: ArrayList(Command) = .{},
    opts: StringHashMap(Value) = .{},
    args: StringHashMap(Value) = .{},

    stdout: *Io.Writer = &stdout_writer.interface,
    stderr: *Io.Writer = &stderr_writer.interface,
    stdin: *Io.Reader = &stdin_reader.interface,

    pub fn help(self: Context) void {
        printHelp(
            self.app,
            &self.path.items[self.path.items.len - 1],
            self.stdout,
        ) catch @panic("Unable to print help text.");
    }
};

pub fn parse(allocator: Allocator, app: *const App) !Context {
    var iter = process.ArgIterator.init();
    _ = iter.skip();

    var ctx = Context{
        .allocator = allocator,
        .app = app,
    };

    var cmd: ?Command = app.root;
    while (cmd) |subcmd| {
        try ctx.path.append(allocator, subcmd);
        cmd = try parseCommand(
            allocator,
            subcmd,
            &iter,
            &ctx.opts,
            &ctx.args,
        );
    }

    return ctx;
}

fn parseCommand(
    allocator: Allocator,
    cmd: Command,
    iter: *process.ArgIterator,
    opts: *StringHashMap(Value),
    args: *StringHashMap(Value),
) !?Command {
    for (cmd.opts) |opt| {
        if (opt.default != null and !opts.contains(opt.name)) {
            try opts.put(allocator, opt.name, opt.default.?);
        }
    }

    for (cmd.args) |arg| {
        if (arg.default != null and !args.contains(arg.name)) {
            try args.put(allocator, arg.name, arg.default.?);
        }
    }

    var positional_count: usize = 0;
    while (iter.next()) |tk| {
        if (isOption(tk)) {
            const is_long = tk.len > 2 and tk[0] == '-' and tk[1] == '-';
            const name = if (is_long) tk[2..] else tk[1..];
            const opt = findOption(cmd, name) orelse
                exit("Unknown option: {s}", .{tk});

            if (opt.flag) {
                try opts.put(allocator, opt.name, .{ .boolean = true });
            } else {
                const val = iter.next() orelse
                    exit("Option {s} is missing a value.", .{tk});

                const parsed = Value.parse(val, opt.kind) catch
                    exit("Unable to parse \"{s}\" of {s} as {}", .{ val, tk, opt.kind });

                try opts.put(allocator, opt.name, parsed);
            }
        } else if (findSubcommand(cmd, tk)) |subcmd| {
            return subcmd;
        } else {
            if (positional_count >= cmd.args.len)
                exit("Unknown argument \"{s}\".", .{tk});

            const arg = cmd.args[positional_count];
            const parsed = Value.parse(tk, arg.kind) catch
                exit("Unable to parse argument \"{s}\" as {}", .{ tk, arg.kind });

            try args.put(allocator, arg.name, parsed);
            positional_count += 1;
        }
    }

    return null;
}

fn isOption(tk: []const u8) bool {
    return tk.len > 1 and std.mem.startsWith(u8, tk, "-") or
        tk.len > 2 and std.mem.startsWith(u8, tk, "--");
}

fn findSubcommand(cmd: Command, name: []const u8) ?Command {
    for (cmd.cmds) |subcmd| {
        if (mem.eql(u8, subcmd.name, name)) {
            return subcmd;
        }
    }
    return null;
}

fn findOption(cmd: Command, name: []const u8) ?Option {
    for (cmd.opts) |opt| {
        if (mem.eql(u8, opt.name, name) or
            opt.long != null and mem.eql(u8, opt.long.?, name) or
            opt.short != null and mem.eql(u8, opt.short.?, name))
        {
            return opt;
        }
    }
    return null;
}

fn exit(comptime fmt: []const u8, args: anytype) noreturn {
    const stderr = &stderr_writer.interface;

    stderr.print(fmt, args) catch @panic("Unable to report error.");
    stderr.writeByte('\n') catch @panic("Unable to report error.");
    stderr.flush() catch @panic("Unable to report error.");

    process.exit(1);
}

fn printHelp(app: *const App, cmd: *const Command, writer: *Writer) !void {
    try writer.print("{s}", .{app.name});

    if (app.version) |ver|
        try writer.print(" {s}", .{ver});

    try writer.writeByte('\n');

    if (app.description) |desc|
        try writer.print("\n{s}\n", .{desc});

    if (cmd.cmds.len > 0) {
        try writer.writeAll("\nSUBCOMMANDS\n");

        for (cmd.cmds) |subcmd| {
            try writer.print("    {s:<20}", .{subcmd.name});

            if (subcmd.description) |desc|
                try writer.print("{s}", .{desc});
        }

        try writer.writeByte('\n');
    }

    if (cmd.args.len > 0) {
        try writer.writeAll("\nARGS\n");

        for (cmd.args) |arg| {
            try writer.print("    {s:<20}", .{arg.name});

            if (arg.description) |desc|
                try writer.print("{s}", .{desc});

            try writer.print(" ({s})", .{@tagName(arg.kind)});
        }

        try writer.writeByte('\n');
    }

    if (cmd.opts.len > 0) {
        try writer.writeAll("\nOPTIONS\n");

        for (cmd.opts) |opt| {
            try writer.writeAll("   ");
            var width: usize = 0;

            if (opt.long != null and opt.short != null) {
                const long = opt.long.?;
                const short = opt.short.?;
                width = long.len + short.len + 3;
                try writer.print("--{s}/-{s}", .{ long, short });
            } else if (opt.long) |long| {
                width = long.len + 2;
                try writer.print("--{s}", .{long});
            } else if (opt.short) |short| {
                width = short.len + 1;
                try writer.print("-{s}", .{short});
            } else {
                width = opt.name.len;
                try writer.print("{s}", .{opt.name});
            }

            _ = try writer.splatByte(' ', 20 - width);

            if (opt.description) |desc|
                try writer.print("{s}", .{desc});

            if (opt.flag) {
                try writer.writeAll(" (flag)");
            } else {
                try writer.print(" ({s})", .{@tagName(opt.kind)});
            }

            try writer.writeByte('\n');
        }

        try writer.writeByte('\n');
    }

    try writer.flush();
}
