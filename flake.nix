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
        # Separate pkgs instance that permits unfree packages (needed for nvidia-x11).
        unfreePkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = commonPackages;
          shellHook = libPathHook;
        };

        # DiariZen requires torch~=2.1 and is not on PyPI (installed from GitHub).
        # torch 2.1.x has no Python 3.13 wheels, so we force Python 3.12 here.
        # This shell uses a separate venv under backends/diarizen/.
        devShells.diarizen = pkgs.mkShell {
          packages = commonPackages;
          shellHook =
            let diarizenLibPath = pkgs.lib.makeLibraryPath [
              pkgs.portaudio
              pkgs.stdenv.cc.cc.lib  # libstdc++.so.6 for pip-installed C extension wheels
              pkgs.zlib               # libz.so.1
              unfreePkgs.linuxPackages.nvidia_x11  # libcuda.so.1 for CUDA-enabled torch wheels
            ];
            in ''
              ${if pkgs.stdenv.isDarwin
                 then "export DYLD_LIBRARY_PATH=${diarizenLibPath}:$DYLD_LIBRARY_PATH"
                 else "export LD_LIBRARY_PATH=${diarizenLibPath}:$LD_LIBRARY_PATH"}
              # uv manages Python 3.11 itself; torch 2.1.x has no cp312/cp313 wheels.
              export UV_PYTHON="cpython-3.11"
              export UV_PROJECT_ENVIRONMENT="$PWD/backends/diarizen/.venv"
            '';
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
