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
      let
        commonPackages = [ pkgs.uv pkgs.just pkgs.portaudio ];
        libPath = pkgs.lib.makeLibraryPath [ pkgs.portaudio ];
        libPathHook = if pkgs.stdenv.isDarwin
          then "export DYLD_LIBRARY_PATH=${libPath}:$DYLD_LIBRARY_PATH"
          else "export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH";
      in
      {
        devShells.default = pkgs.mkShell {
          packages = commonPackages;
          shellHook = libPathHook;
        };

        # Whisperx requires torch~=2.8 which conflicts with torch>=2.10 in the
        # main environment. This shell uses a separate venv under backends/whisperx/.
        devShells.whisperx = pkgs.mkShell {
          packages = commonPackages;
          shellHook = ''
            ${libPathHook}
            export UV_PROJECT_ENVIRONMENT="$PWD/backends/whisperx/.venv"
          '';
        };
      });
}
