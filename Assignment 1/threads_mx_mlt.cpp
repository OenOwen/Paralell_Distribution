#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <omp.h>

// g++ -O2 -march=native -ffast-math -fopenmp threads_mx_mlt.cpp -o thr

// allows us to use cout instead of std::cout every time.
using namespace std;
using namespace std::chrono;

// function to create mx with values in range of size n
// i want the values to be between 1 and 100
vector<vector<int>> createMx(int n)
{
  vector<vector<int>> mx(n, vector<int>(n));
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      mx[i][j] = rand() % 100 + 1;
    }
  }
  return mx;
}

vector<vector<int>> transposeMx(const vector<vector<int>> &mx)
{
    int rows = mx.size(), cols = mx[0].size();
    vector<vector<int>> transposed(cols, vector<int>(rows));

    #pragma omp parallel for // small dffference
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
        transposed[j][i] = mx[i][j];
        }
    }

    return transposed;
}

void printMx(const vector<vector<int>> &mx)
{
  for (const vector<int> &row : mx)
  {
    for (int val : row)
    {
      cout << val << " ";
    }
    cout << "\n";
  }
}

// function that divides mx into 4 matrices 11, 12, 21 and 22
void divideMx(const vector<vector<int>> &mx)
{

  int n = mx.size();

  if (n % 2 != 0)
  {
    cout << "Matrix size must be even\n";
    return;
  }

  if (n < 4)
  {
    cout << "Matrix size must be greater than 8\n";
    return;
  }

  int half_n = n / 2;

  // size of these: half_n x half_n
  vector<vector<int>> mx11(half_n, vector<int>(half_n));
  vector<vector<int>> mx12(half_n, vector<int>(half_n));
  vector<vector<int>> mx21(half_n, vector<int>(half_n));
  vector<vector<int>> mx22(half_n, vector<int>(half_n));



  for (int i = 0; i < half_n; i++)
  {
    for (int j = 0; j < half_n; j++)
    {
      mx11[i][j] = mx[i][j];
      mx12[i][j] = mx[i][j + half_n];
      mx21[i][j] = mx[i + half_n][j];
      mx22[i][j] = mx[i + half_n][j + half_n];
    }
  }

  printMx(mx11);
  cout << "\n";
  printMx(mx12);
  cout << "\n";
  printMx(mx21);
  cout << "\n";
  printMx(mx22);
}

vector<vector<int>> mxMult(const vector<vector<int>> &mx_A,
                           const vector<vector<int>> &mx_B,
                           int n)
{
  vector<vector<int>> mx_C(n, vector<int>(n, 0));

  int **ptr_C = new int*[n];
  for (int i = 0; i < n; ++i)
    ptr_C[i] = mx_C[i].data();


  if (n <= 64)
  {

    vector<vector<int>> transposed_B = transposeMx(mx_B);
    #pragma omp parallel for collapse(2) // no difference
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < n; k++)
        {
          ptr_C[i][j] += mx_A[i][k] * transposed_B[j][k];
        }
      }
    }
  }
  else
  {
    int half_n = n / 2;

    vector<vector<int>> A11(half_n, vector<int>(half_n));
    vector<vector<int>> A12(half_n, vector<int>(half_n));
    vector<vector<int>> A21(half_n, vector<int>(half_n));
    vector<vector<int>> A22(half_n, vector<int>(half_n));

    vector<vector<int>> B11(half_n, vector<int>(half_n));
    vector<vector<int>> B12(half_n, vector<int>(half_n));
    vector<vector<int>> B21(half_n, vector<int>(half_n));
    vector<vector<int>> B22(half_n, vector<int>(half_n));

    // Initialize submatrices
    #pragma omp parallel for collapse(2) // improves speed
    for (int i = 0; i < half_n; i++)
    {
      for (int j = 0; j < half_n; j++)
      {
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

    vector<thread> threads;

    vector<vector<int>> C11_1(half_n, vector<int>(half_n));
    vector<vector<int>> C11_2(half_n, vector<int>(half_n));

    vector<vector<int>> C12_1(half_n, vector<int>(half_n));
    vector<vector<int>> C12_2(half_n, vector<int>(half_n));

    vector<vector<int>> C21_1(half_n, vector<int>(half_n));
    vector<vector<int>> C21_2(half_n, vector<int>(half_n));

    vector<vector<int>> C22_1(half_n, vector<int>(half_n));
    vector<vector<int>> C22_2(half_n, vector<int>(half_n));

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
    
    for (int i = 0; i < half_n; i++)
    {
      for (int j = 0; j < half_n; j++)
      {
        ptr_C[i][j] = C11_1[i][j] + C11_2[i][j];
        ptr_C[i][j + half_n] = C12_1[i][j] + C12_2[i][j];
        ptr_C[i + half_n][j] = C21_1[i][j] + C21_2[i][j];
        ptr_C[i + half_n][j + half_n] = C22_1[i][j] + C22_2[i][j];
      }
    }
  }

  delete[] ptr_C;

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

  vector<vector<int>> A;
  vector<vector<int>> B;

  int totalTime = 0;
  vector<vector<int>> C;
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