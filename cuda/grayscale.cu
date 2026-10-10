// grayscale.cu -- RGB->grayscale, a classic OpenCV op as a 2D CUDA kernel.
// Reads image_rgb.bin (uint8, HxWx3), writes gray_out.bin (float, HxW).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void rgb2gray(const unsigned char* rgb, float* gray, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        int i = (y * W + x) * 3;
        gray[y * W + x] = 0.299f * rgb[i] + 0.587f * rgb[i+1] + 0.114f * rgb[i+2];
    }
}

int main(int argc, char** argv) {
    int W = argc > 1 ? atoi(argv[1]) : 1920;
    int H = argc > 2 ? atoi(argv[2]) : 1080;
    size_t npix = (size_t)W * H;

    unsigned char* hrgb = (unsigned char*)malloc(npix * 3);
    FILE* f = fopen("image_rgb.bin", "rb"); fread(hrgb, 1, npix * 3, f); fclose(f);
    float* hgray = (float*)malloc(npix * sizeof(float));

    unsigned char* drgb; float* dgray;
    cudaMalloc(&drgb, npix * 3); cudaMalloc(&dgray, npix * sizeof(float));
    cudaMemcpy(drgb, hrgb, npix * 3, cudaMemcpyHostToDevice);

    dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16);   // 2D grid
    rgb2gray<<<grd, blk>>>(drgb, dgray, W, H);
    cudaDeviceSynchronize();

    cudaMemcpy(hgray, dgray, npix * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* o = fopen("gray_out.bin", "wb"); fwrite(hgray, sizeof(float), npix, o); fclose(o);
    printf("grayscale wrote gray_out.bin (%dx%d)\n", W, H);

    cudaFree(drgb); cudaFree(dgray); free(hrgb); free(hgray);
    return 0;
}
