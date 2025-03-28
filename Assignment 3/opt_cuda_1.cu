
// $ nvcc -o opt_cuda_1 opt_cuda_1.cu


#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Define matrix size
#define N 1024
#define TILE_SIZE 32    

// CUDA kernel to perform matrix multiplication
__global__ void matrixMult(int *d_A, int *d_B, int *d_C, int n) {

    // Shared memory for tiles
    __shared__ int mxA_shared[TILE_SIZE][TILE_SIZE];
    __shared__ int mxB_shared[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    
    for (int i = 0; i < (n + TILE_SIZE - 1) / TILE_SIZE; i++) {
        mxA_shared[threadIdx.y][threadIdx.x] = d_A[row * n + (i * TILE_SIZE + threadIdx.x)];
        mxB_shared[threadIdx.y][threadIdx.x] = d_B[(i * TILE_SIZE + threadIdx.y) * n + col];


        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++){
            sum += mxA_shared[threadIdx.y][k] * mxB_shared[k][threadIdx.x];
        }

        __syncthreads();;
    }

    if (row < n && col < n) {
        d_C[row * n + col] = sum;
    }
    
}

// Function to create matrix on host
void createMx(int *mx, int n) {
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mx[i * n + j] = rand() % 100 + 1;
        }
    }
}

// Function to create a test matrix used for testing calculation are correct.
void createTestMx(int *mx, int n) {
    int testMatrix[8][8] = {
        {44, 48, 24, 48, 19, 96, 56, 56},
        {47, 82, 54, 50, 80, 65, 39, 44},
        {34, 35, 74, 13, 73, 97, 78, 75},
        {16, 99, 8, 37, 52, 30, 48, 95},
        {77, 72, 94, 47, 19, 1, 2, 65},
        {35, 7, 66, 14, 24, 4, 57, 57},
        {91, 82, 69, 15, 31, 47, 41, 46},
        {97, 48, 82, 48, 29, 30, 43, 5}
    };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mx[i * n + j] = testMatrix[i][j];
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

    // createTestMx(h_A, N);
    // createTestMx(h_B, N);

  
    for (int i = 0; i < 10; i++) {
        // Define grid and block dimensions
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

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


    // printMx(h_A, N);
    // printMx(h_B, N);
    // printMx(h_C, N);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}