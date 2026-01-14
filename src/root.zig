pub const csv = @import("csv.zig");
pub const iris = @import("iris.zig");
pub const layer = @import("layer.zig");
pub const matrix = @import("matrix.zig");
pub const mse = @import("mse.zig");
pub const net = @import("net.zig");
pub const ops = @import("ops.zig");
pub const scale = @import("scale.zig");
pub const score = @import("score.zig");
pub const split = @import("split.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
