// softmax.cu  --  row-wise softmax in CUDA (one block per row, shared-memory reduction).
// Portable by design: uses __syncthreads(), NOT warp-shuffle, so it is wavefront-size agnostic.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define COLS 256   // == blockDim.x, power of two

__global__ void softmax_row(const float* in, float* out, int cols) {
    int row = blockIdx.x;
    int t   = threadIdx.x;
    __shared__ float buf[COLS];

    float v = (t < cols) ? in[row * cols + t] : -1e30f;
    buf[t] = v; __syncthreads();

    // max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) buf[t] = fmaxf(buf[t], buf[t + s]);
        __syncthreads();
    }
    float m = buf[0]; __syncthreads();

    float e = (t < cols) ? expf(v - m) : 0.0f;
    buf[t] = e; __syncthreads();

    // sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) buf[t] += buf[t + s];
        __syncthreads();
    }
    float sum = buf[0];

    if (t < cols) out[row * cols + t] = e / sum;
}

int main() {
    const int rows = 1024, cols = COLS;
    const size_t sz = (size_t)rows * cols * sizeof(float);
    float *hin = (float*)malloc(sz), *hout = (float*)malloc(sz);
    for (int i = 0; i < rows * cols; ++i) hin[i] = ((i * 37) % 100) / 50.0f - 1.0f;

    float *din, *dout;
    cudaMalloc(&din, sz); cudaMalloc(&dout, sz);
    cudaMemcpy(din, hin, sz, cudaMemcpyHostToDevice);

    softmax_row<<<rows, cols>>>(din, dout, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(hout, dout, sz, cudaMemcpyDeviceToHost);
    FILE* g = fopen("softmax_in.bin",  "wb"); fwrite(hin,  sizeof(float), (size_t)rows*cols, g); fclose(g);
    FILE* f = fopen("softmax_out.bin", "wb"); fwrite(hout, sizeof(float), (size_t)rows*cols, f); fclose(f);
    printf("softmax done, wrote softmax_in.bin / softmax_out.bin (%dx%d)\n", rows, cols);

    cudaFree(din); cudaFree(dout); free(hin); free(hout);
    return 0;
}
