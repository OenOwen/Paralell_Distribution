#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <omp.h>

// g++ -O2 -march=native -ffast-math -fopenmp threads_mx_mlt.cpp -o thr

// allows us to use cout instead of std::cout every time.
using namespace std;
using namespace std::chrono;

int n = 1024;

// int** mx_A = new int*[n];
// int** mx_B = new int*[n];

int** mx_A;
int** mx_B;
int** mx_C;
int** transposed_B;

// function to create mx with values in range of size n
// i want the values to be between 1 and 100
int** createMx(int n)
{
  int** mx = new int*[n];
  for (int i = 0; i < n; i++)
  {
    mx[i] = new int[n];
    for (int j = 0; j < n; j++)
    {
      mx[i][j] = rand() % 100 + 1;
    }
  }
  return mx;
}

int** transposeMx(int** mx, int n){
  int** transposed = new int*[n];
  for (int i = 0; i < n; i++) {
      transposed[i] = new int[n]();
  }
  // #pragma omp parallel for
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
          transposed[j][i] = mx[i][j];
      }
  }
  return transposed;
}

void deleteMx(int** mx, int n) {
  for (int i = 0; i < n; i++) {
      delete[] mx[i];
  }
  delete[] mx;
}

void printAMx(int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      cout << mx_A[i][j] << " ";
    }
    cout << "\n";
  }
}

void printBMx(int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      cout << mx_B[i][j] << " ";
    }
    cout << "\n";
  }
}

void printCMx(int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      cout << mx_C[i][j] << " ";
    }
    cout << "\n";
  }
}


void mxMult(int block, int istart, int iend, int jstart, int jend) {
  if (block <= 64)
  {
    #pragma omp parallel for // no difference
    for (int i = istart; i < iend; i++)
    {
      for (int j = jstart; j < jend; j++)
      {
        for (int k = 0; k < n; k++)
        {
          mx_C[i][j] += mx_A[i][k] * mx_B[j][k];
        }
      }
    }

  }
  else {
    int half_block = block / 2;
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        mxMult(half_block, istart, istart + half_block, jstart, jstart + half_block);
      }

      #pragma omp section
      {
        mxMult(half_block, istart, istart + half_block, jstart + half_block, jend);
      }

      #pragma omp section
      {
        mxMult(half_block, istart + half_block, iend, jstart, jstart + half_block);
      }

      #pragma omp section
      {
        mxMult(half_block, istart + half_block, iend, jstart + half_block, jend);
      }
    }
  }


}

int main()
{

  int num_times = 100;

  int totalTime = 0;

  int num_threads = omp_get_max_threads();
  cout << "Using " << num_threads << " threads" << endl;


  for (int i = 0; i < num_times; i++){
    mx_C = new int*[n];
    for (int i = 0; i < n; ++i){
      // mx_A[i] = new int[n]();
      // mx_B[i] = new int[n]();
      mx_C[i] = new int[n]();
    }

    for (int i = 0; i < n; i++){
      fill(mx_C[i], mx_C[i] + n, 0);
    }

    // Filling two matrices A and B, filled with values between 1 and 8
    // for (int i = 0; i < n; ++i) {
    //   for (int j = 0; j < n; ++j) {
    //       mx_A[i][j] = j + 1;
    //       mx_B[i][j] = j + 1;
    //   }
    // }


    // create mx of size n
    mx_A = createMx(n);

    // create mx of size n
    mx_B = createMx(n);
    mx_B = transposeMx(mx_B, n);

  



    auto start = high_resolution_clock::now();
    mxMult(n, 0, n, 0, n);
    auto stop = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(stop - start);
    totalTime += time.count();



    deleteMx(mx_A, n);
    deleteMx(mx_B, n);
    deleteMx(mx_C, n);
  }
  // printAMx(n);
  // cout << "\n";
  // printBMx(n);
  // cout << "\n";
  // printCMx(n);
  

  int time = totalTime / num_times;

  double seconds = time / 1000000.0;

  cout << "\nTime taken by optimized and thredded matrix multiplication: " << time << " microseconds (" << seconds <<" seconds)" << endl;

  return 0;
}