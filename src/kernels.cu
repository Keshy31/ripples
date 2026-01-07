#include <cuda_runtime.h>

extern "C" __global__ void fused_update_kernel(float* prev_u, float* u, float* next_u, float c, float dt, float dx, float damping, float freq, float amp, float t, int source_x, int source_y, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        float lap = (u[(i+1)*size + j] + u[(i-1)*size + j] + 
                     u[i*size + j+1] + u[i*size + j-1] - 
                     4.0f * u[i*size + j]) / (dx * dx);

        float damping_force = damping * dt * (u[i * size + j] - prev_u[i * size + j]);
        float wave_term = (c * c) * (dt * dt) * lap;
        
        next_u[i * size + j] = 2.0f * u[i * size + j] - prev_u[i * size + j] + wave_term - damping_force;
    }

    if (i == source_y && j == source_x) {
        next_u[i * size + j] += amp * sinf(t * 2.0f * 3.1415926535f * freq);
    }
}
