{
  description = "Text to speech experiment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.just
            pkgs.portaudio
          ];
          shellHook =
            let libPath = pkgs.lib.makeLibraryPath [ pkgs.portaudio ];
            in if pkgs.stdenv.isDarwin
               then "export DYLD_LIBRARY_PATH=${libPath}:$DYLD_LIBRARY_PATH"
               else "export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH";
        };
      });
}
