#include <iostream>
#include <vector>
#include <chrono>

// allows us to use cout instead of std::cout every time.
using namespace std;
using namespace std::chrono;


// function to create mx with values in range of size n
// i want the values to be between 1 and 100
vector<vector<int>> createMx(int n) {
  vector<vector<int>> mx(n, vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mx[i][j] = rand() % 100 + 1;
    }
  }
  return mx;
}

vector<vector<int>> transposeMx(const vector<vector<int>> &mx) {
  int rows = mx.size(), cols = mx[0].size();
  vector<vector<int>> transposed(cols, vector<int>(rows));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      transposed[j][i] = mx[i][j];
    }
  }

  return transposed;
}

vector<vector<int>> mxMult(const vector<vector<int>> &mx_A,
                           const vector<vector<int>> &mx_B) {

  int a_rows_cnt = mx_A.size();
  int b_cols_cnt = mx_B[0].size();
  int b_rows_cnt = mx_B.size();

  vector<vector<int>> mx_C(a_rows_cnt, vector<int>(b_cols_cnt, 0));

  vector<vector<int>> transposed_B = transposeMx(mx_B);

  // iter over rows of mx A
  for (int i = 0; i < a_rows_cnt; i++) {

    // iter over cols of mx B
    for (int j = 0; j < b_cols_cnt; j++) {
      int sum = 0;
      // iter over rows in mx B
      for (int k = 0; k < b_rows_cnt; k++) {
        sum += mx_A[i][k] * transposed_B[j][k];        
      }
      mx_C[i][j] = sum;
      //cout << "C[" << i << "][" << j << "] = " << mx_C[i][j] << "\n";
    }
    //cout << "\n";
  }
  return mx_C;
}


void printMx(const vector<vector<int>> &mx) {
  for (const vector<int> &row : mx) {
    for (int val : row) {
      cout << val << " ";
    }
    cout << "\n";
  }
}

int main() {
  //vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  //vector<vector<int>> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
  
  vector<vector<int>> A;
  vector<vector<int>> B;

  int totalTime = 0;
  for (int i = 0; i < 1; i++){

    // create mx of size 25
    A = createMx(1024);

    // create mx of size 25
    B = createMx(1024);

    auto start = high_resolution_clock::now();
    vector<vector<int>> C = mxMult(A, B);
    auto stop = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(stop - start);
    totalTime += time.count();
  }
  

  int time = totalTime / 1;
  
  

  // printMx(A);
  // cout << "x \n";
  // printMx(B);
  // cout << "= \n";
  // printMx(C);

  cout << "\nTime taken by optimized matrix multiplication: "<< time << " microseconds" << endl;

  return 0;
}
