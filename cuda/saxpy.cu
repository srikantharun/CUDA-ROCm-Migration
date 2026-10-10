// saxpy.cu  --  a tiny CUDA program we will migrate to HIP/ROCm.
// Computes y = a*x + y on the GPU, then writes y to saxpy_out.bin.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 20;            // ~1M elements
    const size_t sz = (size_t)n * sizeof(float);
    const float a = 2.0f;

    float *hx = (float*)malloc(sz), *hy = (float*)malloc(sz);
    for (int i = 0; i < n; ++i) { hx[i] = 1.0f; hy[i] = (float)i; }

    float *dx, *dy;
    cudaMalloc(&dx, sz); cudaMalloc(&dy, sz);
    cudaMemcpy(dx, hx, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, sz, cudaMemcpyHostToDevice);

    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    saxpy<<<blocks, tpb>>>(n, a, dx, dy);
    cudaDeviceSynchronize();

    cudaMemcpy(hy, dy, sz, cudaMemcpyDeviceToHost);
    FILE* f = fopen("saxpy_out.bin", "wb");
    fwrite(hy, sizeof(float), n, f); fclose(f);
    printf("saxpy done, wrote saxpy_out.bin (n=%d)\n", n);

    cudaFree(dx); cudaFree(dy); free(hx); free(hy);
    return 0;
}
