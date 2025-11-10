# Cymatics Simulator Technical Overview

## Introduction

This technical overview complements the project overview document, providing in-depth details tailored for the engineering team. It focuses on the mathematical equations, numerical methods, GPU-accelerated simulations using CUDA, and shader-based rendering with OpenGL. The goal is to equip engineers with the precise formulations and code structures needed for implementation, optimization, and extension of the cymatics simulator.

The simulator models vibrational patterns in 2D mediums (e.g., plates or fluids) excited by harmonic sources, rendering them in real-time on an NVIDIA RTX 4080 GPU. Key emphases include accuracy in wave propagation, efficient computation via device arrays, and visually impressive effects through custom GLSL shaders. All components are Zig-based for high performance and minimal overhead, with CUDA integration for GPU acceleration.

Assumptions: The system runs on Zig 0.12+ with NVIDIA CUDA Toolkit, GLFW 3.4+, and GLAD (OpenGL 4.6 loader). Grid resolutions start at 1024x1024 for balance between fidelity and FPS (target: 60+).

(Word count so far: ~150)

## Detailed Theory and Equations

Cymatics simulations rely on solving partial differential equations (PDEs) for wave phenomena. Below, we expand on the core equations, derivations, and boundary conditions.

### The Wave Equation
The foundational model is the 2D scalar wave equation for displacement \(u(x, y, t)\):

\[
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u + f(x, y, t)
\]

where:
- \(c\) is the wave speed (\(\sqrt{T/\rho}\) for membranes, with tension \(T\) and density \(\rho\)),
- \(\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\) is the Laplacian,
- \(f(x, y, t)\) is a forcing term (e.g., sine wave source).

Boundary conditions: Fixed edges (Dirichlet: \(u=0\) at boundaries) for plates, or periodic for infinite domains.

For resonant modes, assume harmonic time dependence \(u(x,y,t) = \phi(x,y) e^{i\omega t}\), reducing to the Helmholtz equation:

\[
\nabla^2 \phi + k^2 \phi = 0, \quad k = \omega / c
\]

Solutions yield eigenmodes and frequencies.

### Chladni Modes for Plates
For square plates (side length \(L\)):

\[
\phi_{n,m}(x,y) = \sin\left(\frac{n\pi x}{L}\right) \sin\left(\frac{m\pi y}{L}\right)
\]

Eigenfrequencies:

\[
f_{n,m} = \frac{c}{2L} \sqrt{n^2 + m^2}
\]

Nodes occur where \(\phi=0\), forming grid-like patterns. For circular plates (radius \(R\)):

\[
\phi_{n,s}(r,\theta) = J_n(k_{n,s} r / R) \cos(n\theta)
\]

where \(J_n\) is the nth-order Bessel function of the first kind, and \(k_{n,s}\) is the sth zero of \(J_n\) (e.g., \(k_{0,1} \approx 2.4048\)). Frequency:

\[
f_{n,s} = \frac{c k_{n,s}}{2\pi R}
\]

These predict radial (\(s\)) and azimuthal (\(n\)) nodes, mimicking bowl setups.

### Faraday Waves in Fluids
For liquids, include gravity \(g\) and surface tension \(\sigma\):

\[
\frac{\partial^2 h}{\partial t^2} + g \nabla^2 h - \frac{\sigma}{\rho} \nabla^4 h = 0
\]

(h: height perturbation). Driven parametrically (vertical oscillation), subharmonic patterns emerge at \(f/2\). Nonlinear terms add chaos:

\[
\frac{\partial^2 h}{\partial t^2} = -g \nabla^2 h + \nu \nabla^2 \frac{\partial h}{\partial t} + \text{nonlinear terms}
\]

(with viscosity \(\nu\)).

### Numerical Considerations
Analytical solutions are limited; use finite differences for time-stepping:

\[
u^{t+1}_{i,j} = 2u^t_{i,j} - u^{t-1}_{i,j} + c^2 \Delta t^2 (\nabla^2 u^t)_{i,j}
\]

Discrete Laplacian (5-point stencil):

\[
(\nabla^2 u)_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{\Delta x^2}
\]

Stability: Courant-Friedrichs-Lewy (CFL) condition \(\frac{c \Delta t}{\Delta x} \leq 1\).

Damping: Add \(-\gamma \frac{\partial u}{\partial t}\) to prevent energy buildup.

(Word count so far: ~650)

## Simulation Methods in Zig

Zig handles the simulation backend, using GPU device pointers for parallel computation via CUDA. This section details array operations and code structures.

### Grid Setup and Initialization
Use 2D slices on host, with CUDA allocations for device:

```zig
const std = @import("std");
const cuda = @cImport(@cInclude("cuda_runtime.h"));

const grid_size: usize = 1024;
const dx: f32 = 1.0 / @as(f32, @floatFromInt(grid_size - 1)); // Normalized domain [0,1]
const c: f32 = 0.1; // Wave speed
const dt: f32 = dx / (c * 1.5); // CFL <1 for stability
const damping: f32 = 0.01;

var device_prev_u: ?*f32 = null;
_ = cuda.cudaMalloc(@as(*?*anyopaque, @ptrCast(&device_prev_u)), grid_size * grid_size * @sizeOf(f32));
var device_u: ?*f32 = null;
_ = cuda.cudaMalloc(@as(*?*anyopaque, @ptrCast(&device_u)), grid_size * grid_size * @sizeOf(f32));
// Initialize to zeros with cudaMemset
```

### Time-Stepping Loop
Implement Verlet integration for the wave equation using CUDA kernels:

```zig
// In kernels.cu (compiled separately)
__global__ void compute_laplacian(float* u, float* lap, int size, float dx) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        lap[i * size + j] = (u[(i+1)*size + j] + u[(i-1)*size + j] + u[i*size + j+1] + u[i*size + j-1] - 4 * u[i*size + j]) / (dx * dx);
    }
}

__global__ void update_step(float* prev_u, float* u, float* next_u, float* lap, float c, float dt, float damping, float freq, float amp, float t, int source_x, int source_y, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        next_u[i * size + j] = 2 * u[i * size + j] - prev_u[i * size + j] + (c * c) * (dt * dt) * lap[i * size + j] - damping * dt * (u[i * size + j] - prev_u[i * size + j]);
    }
    // Source addition (single thread or host-side)
}
```

In Zig, launch kernels:

```zig
fn simulate_step(device_prev_u: *f32, device_u: *f32, device_next_u: *f32, device_lap: *f32, freq: f32, amp: f32, t: f32) !void {
    const block_dim = dim3{ .x = 16, .y = 16, .z = 1 };
    const grid_dim = dim3{ .x = (grid_size + 15) / 16, .y = (grid_size + 15) / 16, .z = 1 };
    compute_laplacian<<<grid_dim, block_dim>>>(device_u, device_lap, grid_size, dx);
    update_step<<<grid_dim, block_dim>>>(device_prev_u, device_u, device_next_u, device_lap, c, dt, damping, freq, amp, t, grid_size / 2, grid_size / 2, grid_size);
    // Swap pointers
    // Boundaries: Handle in kernel or separate
}
```

For modes: Precompute and sum harmonics:

```zig
fn chladni_mode(n: u32, m: u32, x: f32, y: f32) f32 {
    return @sin(@as(f32, @floatFromInt(n)) * std.math.pi * x) * @sin(@as(f32, @floatFromInt(m)) * std.math.pi * y);
}
```

Switch modes via user input.

### Fluid-Specific Extensions
For Faraday waves, add higher-order Laplacians (bi-Laplacian) via nested kernel calls, and parametric driving: Multiply vertical accel in forcing term.

Optimization: Use comptime for constants, or custom PTX if bottlenecks arise. Memory: 1024x1024 float32 array ~4MB; multiple buffers fit in 16GB.

(Word count so far: ~950)

## Rendering and Shaders in OpenGL

OpenGL provides shader-based rendering via GLAD and GLFW. Focus on transferring CUDA data to textures and applying effects.

### Setup and Data Transfer
In main loop:

```zig
const glfw = @cImport(@cInclude("GLFW/glfw3.h"));
const glad = @cImport(@cInclude("glad/glad.h"));

const window = glfw.glfwCreateWindow(1920, 1080, "Cymatics Simulator", null, null) orelse return error.WindowCreationFailed;
// Make context, load GLAD...

// Quad VBO/VAO
const vertices = [_]f32{ -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };
var vbo: glad.GLuint = 0;
glad.glGenBuffers(1, &vbo);
glad.glBindBuffer(glad.GL_ARRAY_BUFFER, vbo);
glad.glBufferData(glad.GL_ARRAY_BUFFER, vertices.len * @sizeOf(f32), &vertices, glad.GL_STATIC_DRAW);

// Texture
var tex: glad.GLuint = 0;
glad.glGenTextures(1, &tex);
glad.glBindTexture(glad.GL_TEXTURE_2D, tex);
glad.glTexImage2D(glad.GL_TEXTURE_2D, 0, glad.GL_R32F, grid_size, grid_size, 0, glad.GL_RED, glad.GL_FLOAT, null);
```

Update texture:

```zig
while (glfw.glfwWindowShouldClose(window) == 0) {
    // Simulate step...
    // For zero-copy: Use CUDA-GL interop
    var pbo: glad.GLuint = 0;
    glad.glGenBuffers(1, &pbo);
    glad.glBindBuffer(glad.GL_PIXEL_UNPACK_BUFFER, pbo);
    // Register with CUDA: cudaGraphicsGLRegisterBuffer
    // Map and copy from device_u
    glad.glBindTexture(glad.GL_TEXTURE_2D, tex);
    glad.glTexSubImage2D(glad.GL_TEXTURE_2D, 0, 0, 0, grid_size, grid_size, glad.GL_RED, glad.GL_FLOAT, null);
    glad.glClear(glad.GL_COLOR_BUFFER_BIT);
    glad.glBindTexture(glad.GL_TEXTURE_2D, tex);
    glad.glDrawArrays(glad.GL_TRIANGLE_STRIP, 0, 4);
    glfw.glfwSwapBuffers(window);
    glfw.glfwPollEvents();
}
```

For zero-copy: Use CUDA-GL interop with `cudaGraphicsGLRegisterBuffer` (reduces latency).

### GLSL Shaders
Vertex shader (simple passthrough):

```glsl
#version 450
in vec2 in_vert;
out vec2 uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    uv = (in_vert + 1.0) / 2.0;
}
```

Fragment shader for basic displacement mapping:

```glsl
#version 450
uniform sampler2D tex;  // Displacement
in vec2 uv;
out vec4 fragColor;
void main() {
    float h = texture(tex, uv).r;
    vec3 color = vec3(0.5 + 0.5 * sin(h * 10.0), 0.5 + 0.5 * cos(h * 5.0), abs(h));
    fragColor = vec4(color, 1.0);
}
```

### Advanced Effects
- **Refraction for Water**: Compute normals from height gradients:

```glsl
vec2 dx = dFdx(uv); vec2 dy = dFdy(uv);
float hx = texture(tex, uv + dx).r - texture(tex, uv - dx).r;
float hy = texture(tex, uv + dy).r - texture(tex, uv - dy).r;
vec3 normal = normalize(vec3(hx, hy, 1.0));
vec2 offset = normal.xy * 0.01;  // Distortion
vec3 bg = texture(backgroundTex, uv + offset).rgb;  // Refract background
fragColor = vec4(bg * (0.5 + 0.5 * normal.z), 1.0);  // Specular
```

- **Bloom/Glow**: Multi-pass: Render to FBO, blur high-intensity, add to base.

Blur shader (Gaussian):

```glsl
uniform sampler2D src;
uniform vec2 dir;  // (1,0) horizontal, (0,1) vertical
void main() {
    vec4 sum = vec4(0);
    float weights[5] = float[](0.06, 0.24, 0.4, 0.24, 0.06);
    for(int i=-2; i<=2; i++) sum += texture(src, uv + float(i)*dir*0.005) * weights[i+2];
    fragColor = sum;
}
```

Composite: `final = base + bloom * intensity`.

- **Particle System for Sand**: Compute shader to advect particles toward nodes:

```glsl
#version 450 core
layout(local_size_x=256) in;
uniform sampler2D heightTex;
buffer Particles { vec4 positions[]; };
void main() {
    uint id = gl_GlobalInvocationID.x;
    vec2 pos = positions[id].xy;
    float h = texture(heightTex, pos).r;
    vec2 grad = ...;  // Compute gradient
    pos -= grad * 0.01 * step(0.1, abs(h));  // Move to low |h|
    positions[id].xy = pos;
}
```

Render as points with instancing.

(Word count so far: ~1350)

## Integration and Optimization

- **Zig-CUDA-OpenGL Bridge**: Use CUDA-GL interop for zero-copy transfers. Profile with NVIDIA Nsight.
- **Performance Tips**: Batch updates, minimize host-device transfers (aim for <1ms/frame). For 4080: Leverage Tensor Cores via custom PTX or mixed-precision (f16).
- **Debugging**: Validate equations with small grids; compare to analytical modes.
- **Scalability**: Extend to 3D with voxel arrays; add ML for inverse problems (e.g., freq from pattern via external bindings if needed).

## References
- Jenny, H. (1967). *Cymatics*.
- Chladni, E. (1787). *Entdeckungen*.
- Zig Documentation: ziglang.org.
- CUDA Toolkit Documentation.
- OpenGL Examples on GitHub.

This document provides the blueprint for engineering implementation. Total word count: approximately 1500. For clarifications, refer to the project overview.