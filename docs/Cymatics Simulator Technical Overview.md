Cymatics Simulator Technical Overview

Introduction

This technical overview complements the project overview document, providing in-depth details tailored for the engineering team. It focuses on the mathematical equations, numerical methods, GPU-accelerated simulations using CUDA, and shader-based rendering with OpenGL.

This revised blueprint prioritizes uncompromised visual fidelity. We will avoid performance shortcuts that degrade image quality (e.g., half-precision compute, 2D-only rendering) and instead leverage the full power of the NVIDIA RTX 4080 GPU to achieve a photorealistic, 4K+ real-time simulation.

Assumptions: The system runs on C++17+ with NVIDIA CUDA Toolkit, GLFW 3.4+, and GLAD (OpenGL 4.6 loader). Grid resolutions will start at 2048x2048 for high-definition simulations, with a target of 60+ FPS at 4K rendering resolution.

(Word count so far: ~150)

Detailed Theory and Equations

The core physics models remain the same, as they are foundational to the simulation.

The Wave Equation

The foundational model is the 2D scalar wave equation for displacement u(x,y,t):
∂t2∂2u​=c2∇2u+f(x,y,t)

where:

    c is the wave speed

    ∇2u=∂x2∂2u​+∂y2∂2u​ is the Laplacian

    f(x,y,t) is a forcing term (e.g., sine wave source)

For resonant modes, we use the Helmholtz equation:
∇2ϕ+k2ϕ=0,k=ω/c

Chladni Modes for Plates

    Square Plates (side length L):
    ϕn,m​(x,y)=sin(Lnπx​)sin(Lmπy​)

    Circular Plates (radius R):
    ϕn,s​(r,θ)=Jn​(kn,s​r/R)cos(nθ)

    where Jn​ is the nth-order Bessel function.

Faraday Waves in Fluids

For high-fidelity liquids, we must include gravity g and surface tension σ:
∂t2∂2h​+g∇2h−ρσ​∇4h=0

(h: height perturbation). The ∇4h (bi-Laplacian) term is crucial for realistic small-scale ripples and will be implemented.

Numerical Considerations

We will use the finite difference method for time-stepping:
ui,jt+1​=2ui,jt​−ui,jt−1​+c2Δt2(∇2ut)i,j​

The discrete Laplacian will use the 5-point stencil.
(∇2u)i,j​=Δx2ui+1,j​+ui−1,j​+ui,j+1​+ui,j−1​−4ui,j​​

Fidelity Mandate: All simulation computations will be performed in full FP32 (float) precision to maintain numerical accuracy and visual stability. FP16 (half-precision) will not be used for the simulation state, as it can introduce unacceptable artifacts.

(Word count so far: ~500)

Simulation Methods in C++ (CUDA)

C++ will manage the simulation backend, using GPU device pointers for parallel computation via CUDA.

Grid Setup and Initialization

We will use a higher-resolution grid as the baseline.
C++

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

const size_t grid_size = 2048; // Increased for fidelity
const float dx = 1.0f / static_cast<float>(grid_size - 1);
const float c = 0.1f;
const float dt = dx / (c * 1.5f); // Stable CFL
const float damping = 0.01f;

float* device_prev_u = nullptr;
float* device_u = nullptr;
float* device_next_u = nullptr; // For triple-buffering the state

// Allocate 3 full grids for prev, current, and next state
cudaMalloc(reinterpret_cast<void**>(&device_prev_u), grid_size * grid_size * sizeof(float));
cudaMalloc(reinterpret_cast<void**>(&device_u), grid_size * grid_size * sizeof(float));
cudaMalloc(reinterpret_cast<void**>(&device_next_u), grid_size * grid_size * sizeof(float));

cudaMemset(device_prev_u, 0, grid_size * grid_size * sizeof(float));
cudaMemset(device_u, 0, grid_size * grid_size * sizeof(float));
cudaMemset(device_next_u, 0, grid_size * grid_size * sizeof(float));

Fused Kernel for Time-Stepping

To maximize performance without sacrificing quality, we will fuse the Laplacian calculation and the update step into a single CUDA kernel. This avoids a slow global memory round-trip for the intermediate Laplacian buffer (device_lap).
C++

// In kernels.cu (compiled separately)
__global__ void fused_update_kernel(float* prev_u, float* u, float* next_u, float c, float dt, float dx, float damping, float freq, float amp, float t, int source_x, int source_y, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard boundaries
    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        // Calculate Laplacian in-place (fused step)
        float lap = (u[(i+1)*size + j] + u[(i-1)*size + j] + 
                     u[i*size + j+1] + u[i*size + j-1] - 
                     4.0f * u[i*size + j]) / (dx * dx);

        // Perform Verlet integration (update step)
        float damping_force = damping * dt * (u[i * size + j] - prev_u[i * size + j]);
        float wave_term = (c * c) * (dt * dt) * lap;
        
        next_u[i * size + j] = 2.0f * u[i * size + j] - prev_u[i * size + j] + wave_term - damping_force;
    }

    // Add source (can be done here or in a separate, small kernel)
    if (i == source_y && j == source_x) {
        next_u[i * size + j] += amp * sinf(t * 2.0f * 3.14159f * freq);
    }
}

In C++, the loop is now simpler, launching one kernel and swapping pointers:
C++

void simulate_step(float* device_prev_u, float* device_u, float* device_next_u, float freq, float amp, float t) {
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim((grid_size + 15) / 16, (grid_size + 15) / 16, 1);
    
    // Launch the single fused kernel
    fused_update_kernel<<<grid_dim, block_dim>>>(device_prev_u, device_u, device_next_u, c, dt, dx, damping, freq, amp, t, grid_size / 2, grid_size / 2, grid_size);

    // Swap pointers for next frame
    float* temp = device_prev_u;
    device_prev_u = device_u;
    device_u = device_next_u;
    device_next_u = temp; 
    // (Note: Pointers must be managed on the host)
}

(Word count so far: ~900)

Rendering and Shaders in OpenGL

This section is heavily revised to focus on 3D geometry and physically-based rendering.

3D Mesh Geometry and Data Transfer

For true visual fidelity, we will not render a simple 2D quad. Instead, we will render a high-density 3D plane mesh (e.g., 256x256 segments) and displace its vertices in the vertex shader using the simulation texture. This creates real 3D geometry, enabling correct lighting, silhouettes, and shadows.

Data Transfer: We will use CUDA-GL Interop for zero-copy texture updates, as this is the highest-performance method and does not impact quality. A Pixel Buffer Object (PBO) will be registered with CUDA, mapped, written to by our kernel, and then unmapped for OpenGL to use as a texture source.

GLSL Shaders (Fidelity-First)

Vertex Shader (with 3D Displacement)

This shader creates the 3D wave geometry.
OpenGL Shading Language

#version 450
layout (location = 0) in vec3 in_pos; // From the 3D plane mesh
out vec2 uv;
out vec3 worldPos;
out vec3 normal;

uniform sampler2D displacementTex; // The 2048x2048 sim texture
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    uv = in_pos.xy * 0.5 + 0.5;
    float h = texture(displacementTex, uv).r;

    // Displace the vertex vertically
    vec3 displacedPos = in_pos + vec3(0.0, 1.0, 0.0) * h * 0.1; // Scale displacement
    worldPos = (model * vec4(displacedPos, 1.0)).xyz;
    gl_Position = projection * view * vec4(worldPos, 1.0);

    // Calculate normal from heightmap for lighting
    float h_L = texture(displacementTex, uv - vec2(1.0/2048.0, 0.0)).r;
    float h_R = texture(displacementTex, uv + vec2(1.0/2048.0, 0.0)).r;
    float h_D = texture(displacementTex, uv - vec2(0.0, 1.0/2048.0)).r;
    float h_U = texture(displacementTex, uv + vec2(0.0, 1.0/2048.0)).r;

    vec3 N = vec3((h_L - h_R) * 0.5, 0.1, (h_D - h_U) * 0.5);
    normal = normalize(mat3(transpose(inverse(model))) * N);
}

Fragment Shader (PBR)

We will replace the simple color logic with a Physically Based Rendering (PBR) workflow for realistic materials (water, sand, metal plate).
OpenGL Shading Language

#version 450
in vec2 uv;
in vec3 worldPos;
in vec3 normal;
out vec4 fragColor;

uniform vec3 camPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

// PBR material properties
uniform vec3 albedo = vec3(0.8, 0.8, 0.9); // Base color
uniform float roughness = 0.1; // 0.0 = mirror, 1.0 = diffuse
uniform float metallic = 0.1; // 0.0 = dielectric, 1.0 = metal

// (PBR lighting functions NDF, G, F ... would be included here)

void main() {
    vec3 N = normalize(normal);
    vec3 V = normalize(camPos - worldPos);
    vec3 L = normalize(lightPos - worldPos);
    
    // (Call PBR functions to get final lit color)
    // float NdotL = ...
    // vec3 specular = ...
    // vec3 diffuse = ...

    // Placeholder for simple Blinn-Phong (replace with full PBR)
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor * albedo;
    
    // For water, add refraction:
    // vec3 R = refract(-V, N, 1.0 / 1.33); // 1.33 = IOR of water
    // vec3 refractedColor = texture(skybox, R).rgb;
    // ...
    
    fragColor = vec4(diffuse, 1.0);
}

Advanced Effects

    Particle System for Sand: The compute shader for advecting particles is a good start. For high fidelity, this must be a 3D particle simulation. Particles must be able to pile up on each other, requiring a 3D grid or spatial hash for collision detection, to form realistic 3D dunes along nodal lines.

    Bloom/Glow: The multi-pass Gaussian blur is effective. We will use a high-quality, 13-tap separable blur and ensure it runs on a full-resolution (4K) offscreen buffer.

    Screen Space Ambient Occlusion (SSAO): With real 3D geometry from the displaced mesh, SSAO is essential. We will add an SSAO post-processing pass to create contact shadows in the troughs of waves and at the base of particle piles, dramatically increasing realism.

(Word count so far: ~1400)

Integration and Fidelity-Focused Optimization

    C++-CUDA-OpenGL Bridge: CUDA-GL interop is mandatory for zero-copy, low-latency data transfer from the simulation (device_u) to the render texture (displacementTex).

    Kernel Optimization: The fused simulation kernel is the single most important performance optimization, as it has zero impact on visual quality.

    Full-Resolution Post-Processing: All post-effects (Bloom, SSAO, FXAA/TAA) will be run at the native 4K target resolution. We will not use dynamic resolution or upscaling unless absolutely necessary.

    Tessellation Shaders: As an advanced alternative to the high-density vertex mesh, hardware Tessellation Shaders can be used to dynamically add geometry based on screen space, providing maximum detail up close without rendering millions of vertices far away.

This document provides the blueprint for an uncompromised, fidelity-first engineering implementation.

(Total word count: ~1500)