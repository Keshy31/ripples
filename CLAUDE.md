# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ripples is a GPU-accelerated cymatics simulator that replicates wave phenomena like Chladni plates and Faraday waves. It uses CUDA for physics simulation and OpenGL for real-time 3D visualization.

## Build Commands

```bash
# Configure and build (from project root)
cmake -B build
cmake --build build

# Run the simulator (must run from project root for shader paths)
./build/ripples      # Linux
.\build\Debug\ripples.exe  # Windows
```

## Dependencies

Dependencies are fetched automatically via CMake FetchContent:
- GLFW 3.4 - Window/input management
- GLM 1.0.1 - Math library
- GLAD - OpenGL loader (bundled with GLFW)
- CUDA Toolkit - Must be installed on system

## Architecture

### Simulation Pipeline

1. **CUDA Kernel** (`src/kernels.cu`): Solves 2D wave equation using Verlet integration with 5-point Laplacian stencil. Triple-buffered (prev_u, u, next_u) for temporal stability.

2. **CUDA-GL Interop** (`src/main.cpp`): Zero-copy transfer from CUDA simulation buffer to OpenGL texture via `cudaGraphicsGLRegisterImage`.

3. **OpenGL Rendering** (`src/shaders/`): Displacement-mapped 3D mesh (256x256 segments) with Phong lighting.

### Key Parameters (in main.cpp)

- `grid_size = 2048`: Simulation resolution
- `c = 0.1`: Wave speed
- `dt = dx / (c * 1.5)`: Time step (CFL-stable)
- `damping = 0.01`: Damping coefficient
- `freq = 10.0`, `amp = 0.1`: Source oscillation

### Shader Paths

Shaders are loaded relative to working directory: `src/shaders/vertex.glsl` and `src/shaders/fragment.glsl`. The executable must be run from the project root.

## Physics Reference

The simulator implements the 2D scalar wave equation:
```
∂²u/∂t² = c²∇²u + f(x,y,t)
```

See `docs/Cymatics Simulator Technical Overview.md` for detailed equations and numerical methods.
