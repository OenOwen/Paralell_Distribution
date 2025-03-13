#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <omp.h>

// g++ -O2 -march=native -ffast-math threads_mx_mlt.cpp -o thr

// allows us to use cout instead of std::cout every time.
using namespace std;
using namespace std::chrono;

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
      transposed[i] = new int[n];
  }
  #pragma omp parallel for
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

void printMx(int** mx, int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      cout << mx[i][j] << " ";
    }
    cout << "\n";
  }
}


int** mxMult(int** mx_A, int** mx_B, int n)
{

  int **mx_C = new int*[n];
  for (int i = 0; i < n; ++i){
    mx_C[i] = new int[n]();
  }
  int** transposed_B = transposeMx(mx_B, n);

  if (n <= 64)
  {

    #pragma omp parallel for collapse(2) // no difference
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < n; k++)
        {
          mx_C[i][j] += mx_A[i][k] * transposed_B[j][k];
        }
      }
    }
  }
  else
  {
    int half_n = n / 2;
    int** A11 = new int*[half_n];
    int** A12 = new int*[half_n];
    int** A21 = new int*[half_n];
    int** A22 = new int*[half_n];

    int** B11 = new int*[half_n];
    int** B12 = new int*[half_n];
    int** B21 = new int*[half_n];
    int** B22 = new int*[half_n];

    for (int i = 0; i < half_n; i++) {
        A11[i] = new int[half_n]; A12[i] = new int[half_n];
        A21[i] = new int[half_n]; A22[i] = new int[half_n];
        B11[i] = new int[half_n]; B12[i] = new int[half_n];
        B21[i] = new int[half_n]; B22[i] = new int[half_n];
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < half_n; i++) {
      for (int j = 0; j < half_n; j++) {
        A11[i][j] = mx_A[i][j];
        A12[i][j] = mx_A[i][j + half_n];
        A21[i][j] = mx_A[i + half_n][j];
        A22[i][j] = mx_A[i + half_n][j + half_n];

        B11[i][j] = mx_B[i][j];
        B12[i][j] = mx_B[i][j + half_n];
        B21[i][j] = mx_B[i + half_n][j];
        B22[i][j] = mx_B[i + half_n][j + half_n];
      }
    }
    int** C11_1 = new int*[half_n];
    int** C11_2 = new int*[half_n];
    int** C12_1 = new int*[half_n];
    int** C12_2 = new int*[half_n];

    int** C21_1 = new int*[half_n];
    int** C21_2 = new int*[half_n];
    int** C22_1 = new int*[half_n];
    int** C22_2 = new int*[half_n];

    #pragma omp parallel sections
    {
      #pragma omp section
      C11_1 = mxMult(A11, B11, half_n);

      #pragma omp section
      C11_2 = mxMult(A12, B21, half_n);

      #pragma omp section
      C12_1 = mxMult(A11, B12, half_n);

      #pragma omp section
      C12_2 = mxMult(A12, B22, half_n);

      #pragma omp section
      C21_1 = mxMult(A21, B11, half_n);

      #pragma omp section
      C21_2 = mxMult(A22, B21, half_n);

      #pragma omp section
      C22_1 = mxMult(A21, B12, half_n);

      #pragma omp section
      C22_2 = mxMult(A22, B22, half_n);
    }


    for (int i = 0; i < half_n; i++) {
      for (int j = 0; j < half_n; j++) {
        mx_C[i][j] = C11_1[i][j] + C11_2[i][j];
        mx_C[i][j + half_n] = C12_1[i][j] + C12_2[i][j];
        mx_C[i + half_n][j] = C21_1[i][j] + C21_2[i][j];
        mx_C[i + half_n][j + half_n] = C22_1[i][j] + C22_2[i][j];
      }
    }

    deleteMx(A11, half_n); deleteMx(A12, half_n);
    deleteMx(A21, half_n); deleteMx(A22, half_n);
    deleteMx(B11, half_n); deleteMx(B12, half_n);
    deleteMx(B21, half_n); deleteMx(B22, half_n);
    deleteMx(C11_1, half_n); deleteMx(C11_2, half_n);
    deleteMx(C12_1, half_n); deleteMx(C12_2, half_n);
    deleteMx(C21_1, half_n); deleteMx(C21_2, half_n);
    deleteMx(C22_1, half_n); deleteMx(C22_2, half_n);
  }

  return mx_C;
}

int main()
{
//   vector<vector<int>> A = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
//                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};

  // // 8x8 matrix
  // vector<vector<int>> B = {{1, 2, 3, 4, 5, 6, 7, 8},
  //                           {1, 2, 3, 4, 5, 6, 7, 8},
  //                           {1, 2, 3, 4, 5, 6, 7, 8},
  //                           {1, 2, 3, 4, 5, 6, 7, 8},
  //                           {1, 2, 3, 4, 5, 6, 7, 8},
  //                           {1, 2, 3, 4, 5, 6, 7, 8},
  //                           {1, 2, 3, 4, 5, 6, 7, 8},
  //                           {1, 2, 3, 4, 5, 6, 7, 8}};

  // // 4x4 matrix D
  // vector<vector<int>> D = {{1, 2, 3, 4},
  //                           {1, 2, 3, 4},
  //                           {1, 2, 3, 4},
  //                           {1, 2, 3, 4}};

  int** A;
  int** B;
  int** C;

  int totalTime = 0;
  for (int i = 0; i < 10; i++){

    // create mx of size 25
    A = createMx(1024);

    // create mx of size 25
    B = createMx(1024);

    auto start = high_resolution_clock::now();
    C = mxMult(A, B, 1024);
    auto stop = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(stop - start);
    totalTime += time.count();
  }

//   printMx(C);
  

  int time = totalTime / 10;
  
  

  

  // printMx(A);
  // cout << "x \n";
  // printMx(B);
  // cout << "= \n";

  double seconds = time / 1000000.0;

  cout << "\nTime taken by optimized matrix multiplication: " << time << " microseconds (" << seconds <<" seconds)" << endl;

  return 0;
}