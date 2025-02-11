#include <iostream>
#include <vector>

// allows us to use cout instead of std::cout every time.
using namespace std;

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
  vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  vector<vector<int>> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

  vector<vector<int>> C = mxMult(A, B);

  printMx(A);
  cout << "x \n";
  printMx(B);
  cout << "= \n";
  printMx(C);

  return 0;
}
