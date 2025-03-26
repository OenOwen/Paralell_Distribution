
// nvcc -o cda 3loop_cuda.cu


#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Define matrix size
#define N 1024

// CUDA kernel to perform matrix multiplication
__global__ void matrixMult(int *d_A, int *d_B, int *d_C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += d_A[row * n + k] * d_B[k * n + col];
        }
        d_C[row * n + col] = sum;
    }
}

// Function to create matrix on host
void createMx(int *mx, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mx[i * n + j] = i + j;
        }
    }
}

void printMx(int *mx, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << mx[i * n + j] << " ";
        }
        cout << "\n";
    }
}

int main() {
    const int size = N * N * sizeof(int);
    float totalTime = 0.0f;

    // Allocate host memory for matrices
    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];

    // Allocate device memory for matrices
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Initialize matrices on host
    createMx(h_A, N);
    createMx(h_B, N);

    for (int i = 0; i < 10; i++) {
        // Define grid and block dimensions
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

        // Copy data from host to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start time
        cudaEventRecord(start, 0);

        // Launch CUDA kernel to multiply matrices
        matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Record stop time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        cudaEventElapsedTime(&elapsedTime, start, stop);
        totalTime += elapsedTime;

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate average time
    float avgTime = totalTime / 10.0f;

    cout << "\nTime taken for 3loop Matrix multiplication (CUDA): " 
         << avgTime << " milliseconds (" 
         << avgTime / 1000.0f << " seconds)" << endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //printMx(h_C, 1024);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}