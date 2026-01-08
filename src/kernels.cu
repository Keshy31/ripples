#include <cuda_runtime.h>

// Maximum number of sources
#define MAX_SOURCES 8

// Pi constant
#define PI 3.1415926535f

// Simulation modes
enum SimMode {
    SIM_WAVE_EQUATION = 0,  // Standard 2D wave equation
    SIM_FARADAY_WAVES = 1,  // Faraday waves with surface tension
    SIM_CHLADNI_MODES = 2   // Chladni plate modes (Bessel functions)
};

// Source structure for wave excitation
struct Source {
    int x, y;           // Grid position
    float freq;         // Oscillation frequency (Hz)
    float amp;          // Amplitude
    float phase;        // Phase offset (radians)
    int active;         // 1 = active, 0 = inactive
};

// Faraday wave parameters
struct FaradayParams {
    float surface_tension;  // σ (sigma) - surface tension coefficient
    float gravity;          // g - gravitational acceleration
    float density;          // ρ (rho) - fluid density
    float drive_freq;       // Parametric driving frequency (Hz)
    float drive_amp;        // Driving amplitude (modulation depth 0-1)
};

// Chladni mode parameters
struct ChladniParams {
    int mode_n;             // Azimuthal mode number (0-5)
    int mode_s;             // Radial mode number (1-5)
    float freq;             // Driving frequency
    float amp;              // Amplitude
    int circular_mask;      // 1 = apply circular boundary, 0 = square
};

// Bessel function zeros (kn,s values) for modes (n=0-5, s=1-5)
// These are the sth zeros of Jn(x) = 0
__constant__ float BESSEL_ZEROS[6][5] = {
    {2.4048f, 5.5201f, 8.6537f, 11.7915f, 14.9309f},   // J0 zeros
    {3.8317f, 7.0156f, 10.1735f, 13.3237f, 16.4706f},  // J1 zeros
    {5.1356f, 8.4172f, 11.6198f, 14.7960f, 17.9598f},  // J2 zeros
    {6.3802f, 9.7610f, 13.0152f, 16.2235f, 19.4094f},  // J3 zeros
    {7.5883f, 11.0647f, 14.3725f, 17.6160f, 20.8269f}, // J4 zeros
    {8.7715f, 12.3386f, 15.7002f, 18.9801f, 22.2178f}  // J5 zeros
};

// Bessel function J0(x) approximation (polynomial for |x| < 8, asymptotic otherwise)
__device__ float bessel_j0(float x) {
    x = fabsf(x);
    if (x < 8.0f) {
        float y = x * x;
        return 1.0f - y * (0.25f - y * (0.015625f - y * (0.000434028f - y * 0.0000067816f)));
    } else {
        float z = 8.0f / x;
        float y = z * z;
        float theta = x - 0.785398163f;  // x - pi/4
        return sqrtf(0.636619772f / x) * cosf(theta);
    }
}

// Bessel function J1(x) approximation
__device__ float bessel_j1(float x) {
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    x = fabsf(x);
    if (x < 8.0f) {
        float y = x * x;
        return sign * x * (0.5f - y * (0.0625f - y * (0.00260417f - y * 0.0000542535f)));
    } else {
        float z = 8.0f / x;
        float theta = x - 2.356194491f;  // x - 3*pi/4
        return sign * sqrtf(0.636619772f / x) * cosf(theta);
    }
}

// General Bessel function Jn(x) using recurrence relation
// Jn+1(x) = (2n/x) * Jn(x) - Jn-1(x)
__device__ float bessel_jn(int n, float x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);

    // For small x, use series expansion for small n
    if (fabsf(x) < 0.01f) {
        // Jn(x) ≈ (x/2)^n / n! for small x
        float result = 1.0f;
        float half_x = x * 0.5f;
        for (int i = 0; i < n; i++) {
            result *= half_x / (i + 1);
        }
        return result;
    }

    // Downward recurrence for stability
    float j_prev = bessel_j0(x);
    float j_curr = bessel_j1(x);

    for (int k = 1; k < n; k++) {
        float j_next = (2.0f * k / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }

    return j_curr;
}

// Compute bi-Laplacian using 13-point stencil
__device__ float compute_bilaplacian(float* u, int i, int j, int size, float dx) {
    int idx = i * size + j;
    float dx4 = dx * dx * dx * dx;

    if (i < 2 || i >= size - 2 || j < 2 || j >= size - 2) {
        return 0.0f;
    }

    float bilap =
        20.0f * u[idx]
        - 8.0f * u[(i+1)*size + j]
        - 8.0f * u[(i-1)*size + j]
        - 8.0f * u[i*size + j+1]
        - 8.0f * u[i*size + j-1]
        + 2.0f * u[(i+1)*size + j+1]
        + 2.0f * u[(i+1)*size + j-1]
        + 2.0f * u[(i-1)*size + j+1]
        + 2.0f * u[(i-1)*size + j-1]
        + 1.0f * u[(i+2)*size + j]
        + 1.0f * u[(i-2)*size + j]
        + 1.0f * u[i*size + j+2]
        + 1.0f * u[i*size + j-2];

    return bilap / dx4;
}

// Main wave equation solver with multi-source, Faraday, and Chladni support
extern "C" __global__ void fused_update_kernel(
    float* prev_u, float* u, float* next_u,
    float c, float dt, float dx, float damping,
    Source* sources, int num_sources,
    int sim_mode, FaradayParams faraday, ChladniParams chladni,
    float t, int size
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * size + j;

    // Grid center and radius for circular domain
    float cx = size * 0.5f;
    float cy = size * 0.5f;
    float R = size * 0.5f - 2.0f;  // Slightly smaller to avoid boundary issues

    // Compute polar coordinates from center
    float dx_pos = j - cx;
    float dy_pos = i - cy;
    float r = sqrtf(dx_pos * dx_pos + dy_pos * dy_pos);
    float theta = atan2f(dy_pos, dx_pos);

    // Check if inside circular domain (for Chladni mode)
    bool inside_circle = (r < R);

    // Guard boundaries
    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        // For Chladni mode with circular mask, zero outside circle
        if (sim_mode == SIM_CHLADNI_MODES && chladni.circular_mask && !inside_circle) {
            next_u[idx] = 0.0f;
            return;
        }

        // Calculate Laplacian using 5-point stencil
        float lap = (u[(i+1)*size + j] + u[(i-1)*size + j] +
                     u[i*size + j+1] + u[i*size + j-1] -
                     4.0f * u[idx]) / (dx * dx);

        // Damping force (velocity-proportional)
        float damping_force = damping * dt * (u[idx] - prev_u[idx]);

        // Physics term depends on simulation mode
        float physics_term;

        if (sim_mode == SIM_FARADAY_WAVES) {
            // Faraday-like standing wave patterns via structured forcing
            // Instead of unstable parametric modulation, use direct oscillating forcing
            // with a spatial pattern that creates interference -> standing waves

            // Standard wave equation (stable)
            float c_base = faraday.gravity;
            physics_term = (dt * dt) * c_base * c_base * lap;

            // Create standing wave pattern via grid of virtual sources
            // This mimics the hexagonal/square patterns seen in real Faraday waves
            float pattern_scale = faraday.surface_tension * 100.0f + 50.0f;  // Controls pattern size

            // Hexagonal lattice forcing (creates honeycomb-like patterns)
            float px = (float)j / pattern_scale;
            float py = (float)i / pattern_scale;

            // Three wave directions at 60° angles (creates hexagonal symmetry)
            float k1 = cosf(2.0f * PI * px);
            float k2 = cosf(2.0f * PI * (0.5f * px + 0.866f * py));
            float k3 = cosf(2.0f * PI * (0.5f * px - 0.866f * py));

            // Combine for hexagonal pattern, modulated by drive frequency
            float spatial_pattern = (k1 + k2 + k3) / 3.0f;
            float temporal = sinf(2.0f * PI * faraday.drive_freq * t);

            // Apply forcing
            float forcing = faraday.drive_amp * spatial_pattern * temporal;
            physics_term += forcing * dt * dt;

            // Moderate damping to reach steady state
            damping_force = 0.02f * (u[idx] - prev_u[idx]);
        } else {
            // Standard wave equation (used for both WAVE_EQUATION and CHLADNI_MODES)
            physics_term = (c * c) * (dt * dt) * lap;
        }

        // Verlet integration
        next_u[idx] = 2.0f * u[idx] - prev_u[idx] + physics_term - damping_force;

        // For Chladni mode, drive from center to excite natural modes
        if (sim_mode == SIM_CHLADNI_MODES) {
            // Apply fixed (zero) boundary at circle edge for standing wave formation
            if (!inside_circle) {
                next_u[idx] = 0.0f;
            }
            // Drive from center region - this excites the natural modes
            else if (r < R * 0.05f) {
                float driving = chladni.amp * sinf(2.0f * PI * chladni.freq * t);
                next_u[idx] += driving;
            }
        }
    }

    // Enforce circular boundary for Chladni (zero displacement at edge)
    if (sim_mode == SIM_CHLADNI_MODES && chladni.circular_mask && !inside_circle) {
        next_u[idx] = 0.0f;
    }

    // Apply point sources (wave mode only)
    if (sim_mode == SIM_WAVE_EQUATION) {
        for (int s = 0; s < num_sources && s < MAX_SOURCES; s++) {
            if (sources[s].active && i == sources[s].y && j == sources[s].x) {
                float forcing = sources[s].amp * sinf(t * 2.0f * PI * sources[s].freq + sources[s].phase);
                next_u[idx] += forcing;
            }
        }
    }
}
