#include <iostream>
#include <vector>
#include <chrono>

// allows us to use cout instead of std::cout every time.
using namespace std;
using namespace std::chrono;

vector<vector<int>> createMx(int n) {
  vector<vector<int>> mx(n, vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mx[i][j] = i + j;
    }
  }
  return mx;
}

vector<vector<int>> mxMult(const vector<vector<int>> &mx_A,
                           const vector<vector<int>> &mx_B) {

  int a_rows_cnt = mx_A.size();
  int b_cols_cnt = mx_B[0].size();
  int b_rows_cnt = mx_B.size();

  vector<vector<int>> mx_C(a_rows_cnt, vector<int>(b_cols_cnt, 0));

  // iter over rows of mx A
  for (int i = 0; i < a_rows_cnt; i++) {

    // iter over cols of mx B
    for (int j = 0; j < b_cols_cnt; j++) {

      // iter over rows in mx B
      for (int k = 0; k < b_rows_cnt; k++) {
        int valToAdd = mx_A[i][k] * mx_B[k][j];
        mx_C[i][j] += valToAdd;
      }

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
  

  int totalTime = 0;
  for (int i = 0; i < 10; i++){
    vector<vector<int>> A = createMx(1024);
    vector<vector<int>> B = createMx(1024);
    auto start = high_resolution_clock::now();
    vector<vector<int>> C = mxMult(A, B);
    auto stop = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(stop - start);
    totalTime += time.count();
  }
  

  int time = totalTime / 10;

  // printMx(A);
  // cout << "x \n";
  // printMx(B);
  // cout << "= \n";
  // printMx(C);

  float seconds = time / 1000000.0;

  cout << "\nTime taken for 3loop Matrix multiplication: " << time << " microseconds (" << seconds <<" seconds)" << endl;
  return 0;
}
