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
        # ffmpeg_7 is added because torchcodec 0.7 (compatible with torch 2.8) only
        # supports FFmpeg 4-7, while the system ships FFmpeg 8.
        devShells.whisperx = pkgs.mkShell {
          packages = commonPackages ++ [ pkgs.ffmpeg_7 ];
          shellHook =
            let whisperxLibPath = pkgs.lib.makeLibraryPath [ pkgs.portaudio pkgs.ffmpeg_7 ];
            in ''
              ${if pkgs.stdenv.isDarwin
                 then "export DYLD_LIBRARY_PATH=${whisperxLibPath}:$DYLD_LIBRARY_PATH"
                 else "export LD_LIBRARY_PATH=${whisperxLibPath}:$LD_LIBRARY_PATH"}
              export UV_PROJECT_ENVIRONMENT="$PWD/backends/whisperx/.venv"
            '';
        };
      });
}
