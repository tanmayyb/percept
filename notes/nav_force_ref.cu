#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

struct Obstacle {
    float cx, cy, cz, radius;
};

struct World {
    float cx, cy, cz, radius;
};

struct Goal {
    float x, y, z;
};

// 3D distance
__device__ __host__ float distance3(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ float calculate_gamma(float x, float y, float z, const Goal& goal, int K) {
    float dist = distance3(x, y, z, goal.x, goal.y, goal.z);
    return powf(dist, 2 * K);
}

__device__ float calculate_beta(float x, float y, float z, const World& world, 
                               const Obstacle* obstacles, int n_obstacles) {
    float wc = distance3(x, y, z, world.cx, world.cy, world.cz);
    float beta = world.radius * world.radius - wc * wc;
    for (int i = 0; i < n_obstacles; ++i) {
        float d = distance3(x, y, z, obstacles[i].cx, obstacles[i].cy, obstacles[i].cz);
        beta *= (d * d - obstacles[i].radius * obstacles[i].radius);
    }
    return beta;
}

__device__ float calculate_alpha(float x, float y, float z, const Goal& goal, 
                                 const World& world, const Obstacle* obstacles, int n_obstacles, int K) {
    float gamma = calculate_gamma(x, y, z, goal, K);
    float beta = calculate_beta(x, y, z, world, obstacles, n_obstacles);
    return gamma / beta;
}

__device__ float calculate_phi(float x, float y, float z, const Goal& goal, 
                               const World& world, const Obstacle* obstacles, int n_obstacles, int K) {
    float alpha = calculate_alpha(x, y, z, goal, world, obstacles, n_obstacles, K);
    if (alpha < 0) return 1.0f;
    return powf(alpha / (1.0f + alpha), 1.0f / K);
}

// Numerical gradient, central differences, parallelized over batch
__global__ void batch_force_kernel(
    const float* x, const float* y, const float* z,
    float* fx, float* fy, float* fz,
    int N, Goal goal, World world, const Obstacle* obstacles, int n_obstacles, int K, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx], py = y[idx], pz = z[idx];
    // Central finite differences for gradient (force = -grad(phi))
    float phi0 = calculate_phi(px, py, pz, goal, world, obstacles, n_obstacles, K);
    float phixp = calculate_phi(px + eps, py, pz, goal, world, obstacles, n_obstacles, K);
    float phixm = calculate_phi(px - eps, py, pz, goal, world, obstacles, n_obstacles, K);
    float phix = (phixp - phixm) / (2 * eps);

    float phiyp = calculate_phi(px, py + eps, pz, goal, world, obstacles, n_obstacles, K);
    float phiym = calculate_phi(px, py - eps, pz, goal, world, obstacles, n_obstacles, K);
    float phiy = (phiyp - phiym) / (2 * eps);

    float phizp = calculate_phi(px, py, pz + eps, goal, world, obstacles, n_obstacles, K);
    float phizm = calculate_phi(px, py, pz - eps, goal, world, obstacles, n_obstacles, K);
    float phiz = (phizp - phizm) / (2 * eps);

    fx[idx] = -phix;
    fy[idx] = -phiy;
    fz[idx] = -phiz;
}

// Host-side function
void get_navigation_forces(
    const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& z,
    std::vector<float>& fx, std::vector<float>& fy, std::vector<float>& fz,
    const Goal& goal, const World& world, const std::vector<Obstacle>& obstacles, int K)
{
    int N = x.size();
    float *d_x, *d_y, *d_z, *d_fx, *d_fy, *d_fz;
    Obstacle* d_obstacles;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_fx, N * sizeof(float));
    cudaMalloc(&d_fy, N * sizeof(float));
    cudaMalloc(&d_fz, N * sizeof(float));
    cudaMalloc(&d_obstacles, obstacles.size() * sizeof(Obstacle));
    cudaMemcpy(d_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacles, obstacles.data(), obstacles.size() * sizeof(Obstacle), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (N + threads - 1) / threads;
    float eps = 1e-3f;
    batch_force_kernel<<<blocks, threads>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz, N,
        goal, world, d_obstacles, obstacles.size(), K, eps);

    fx.resize(N); fy.resize(N); fz.resize(N);
    cudaMemcpy(fx.data(), d_fx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), d_fy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), d_fz, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    cudaFree(d_obstacles);
}

// Example usage:
int main() {
    int K = 3;
    Goal goal = {15.0f, 0.0f, 0.0f};
    World world = {0.0f, 0.0f, 0.0f, 20.0f};
    std::vector<Obstacle> obstacles = {
        {-8.0f, 5.0f, 0.0f, 5.0f},
        {0.0f, -9.0f, 0.0f, 4.0f},
        {5.0f, 0.0f, 0.0f, 4.0f}
    };
    std::vector<float> x = {0.0f}, y = {0.0f}, z = {0.0f};
    std::vector<float> fx, fy, fz;
    get_navigation_forces(x, y, z, fx, fy, fz, goal, world, obstacles, K);
    std::cout << "Force at (0,0,0): (" << fx[0] << ", " << fy[0] << ", " << fz[0] << ")\n";
    return 0;
}