{
  description = "A toy library for building neural networks.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let pkgs = nixpkgs.legacyPackages.x86_64-linux;
        deps = [
          pkgs.zig
          pkgs.zls
          pkgs.just
          pkgs.curl
        ];
    in {
      devShell.x86_64-linux = pkgs.mkShell {
        buildInputs = deps;
      };
    };
}
