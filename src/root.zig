pub const csv = @import("csv.zig");
pub const dataset = @import("dataset.zig");
pub const mse = @import("mse.zig");
pub const net = @import("net.zig");
pub const scale = @import("scale.zig");
pub const score = @import("score.zig");
pub const split = @import("split.zig");
pub const tensor = @import("tensor.zig");
pub const matrix = @import("matrix.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
