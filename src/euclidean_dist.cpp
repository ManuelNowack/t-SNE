#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/debug.h>
#include <tsne/matrix.h>

/*
* Unroll inner-most loop by 2.
*/
void euclidean_dist_unroll2(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // calculate non-diagonal entries
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Euclidean distance
      double sum0 = 0;
      double sum1 = 0;
      int k=0;
      for (; k < m-1; k += 2) {
        double dist0 = X->data[i * m + k] - X->data[j * m + k];
        sum0 += dist0 * dist0;
        double dist1 = X->data[i * m + k + 1] - X->data[j * m + k + 1];
        sum1 += dist1 * dist1;
      }
      for (; k < m; k++) {
        double dist0 = X->data[i * m + k] - X->data[j * m + k];
        sum0 += dist0 * dist0;
      }
      double sum = sum0 + sum1;
      D->data[i * n + j] = sum;
      D->data[j * n + i] = sum;
    }
  }

  // set diagonal entries
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Unroll inner-most loop by 4.
*/
void euclidean_dist_unroll4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // calculate non-diagonal entries
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Euclidean distance
      double sum0 = 0;
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      int k=0;
      for (; k < m-3; k += 4) {
        double dist0 = X->data[i * m + k] - X->data[j * m + k];
        sum0 += dist0 * dist0;
        double dist1 = X->data[i * m + k + 1] - X->data[j * m + k + 1];
        sum1 += dist1 * dist1;
        double dist2 = X->data[i * m + k + 2] - X->data[j * m + k + 2];
        sum2 += dist2 * dist2;
        double dist3 = X->data[i * m + k + 3] - X->data[j * m + k + 3];
        sum3 += dist3 * dist3;
      }
      for (; k < m; k++) {
        double dist0 = X->data[i * m + k] - X->data[j * m + k];
        sum0 += dist0 * dist0;
      }
      double sum = (sum0 + sum1) + (sum2 + sum3);
      D->data[i * n + j] = sum;
      D->data[j * n + i] = sum;
    }
  }

  // set diagonal entries
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Unroll inner-most loop by 8.
*/
void euclidean_dist_unroll8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // calculate non-diagonal entries
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Euclidean distance
      double sums[8] = {0};
      double dist;
      int k=0;
      for (; k < m-7; k += 8) {
        dist = X->data[i * m + k] - X->data[j * m + k];
        sums[0] += dist * dist;
        dist = X->data[i * m + k + 1] - X->data[j * m + k + 1];
        sums[1] += dist * dist;
        dist = X->data[i * m + k + 2] - X->data[j * m + k + 2];
        sums[2] += dist * dist;
        dist = X->data[i * m + k + 3] - X->data[j * m + k + 3];
        sums[3] += dist * dist;
        dist = X->data[i * m + k + 4] - X->data[j * m + k + 4];
        sums[4] += dist * dist;
        dist = X->data[i * m + k + 5] - X->data[j * m + k + 5];
        sums[5] += dist * dist;
        dist = X->data[i * m + k + 6] - X->data[j * m + k + 6];
        sums[6] += dist * dist;
        dist = X->data[i * m + k + 7] - X->data[j * m + k + 7];
        sums[7] += dist * dist;
      }
      for (; k < m; k++) {
        double dist = X->data[i * m + k] - X->data[j * m + k];
        sums[0] += dist * dist;
      }
      double sum04 = (sums[0] + sums[1]) + (sums[2] + sums[3]);
      double sum48 = (sums[4] + sums[5]) + (sums[6] + sums[7]);
      double sum = sum04 + sum48;
      D->data[i * n + j] = sum;
      D->data[j * n + i] = sum;
    }
  }

  // set diagonal entries
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Blocking for point dimensions
*/
void euclidean_dist_block8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // initialize distances of upper diagonal elements to 0
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      D->data[i * n + j] = 0;
    }
  }

  int block_size = 4*8;
  // calculate non-diagonal entries
  int k = 0;
  for (; k < (m/block_size)*block_size; k+=block_size) {
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        // Euclidean distance
        double sum = 0;
        for (int kk = k; kk < k+block_size; kk++) {
          double dist = X->data[i * m + kk] - X->data[j * m + kk];
          sum += dist * dist;
        }
        D->data[i * n + j] += sum;
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Euclidean distance
      double sum = 0;
      for (int kk = k; kk < m; kk++) {
        double dist = X->data[i * m + kk] - X->data[j * m + kk];
        sum += dist * dist;
      }
      D->data[i * n + j] += sum;
    }
  }

  // set distances of lower diagonal elements
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      D->data[i * n + j] = D->data[j * n + i];
    }
  }
}

/*
* Blocking for point dimensions and points
*/
void euclidean_dist_block8x8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // initialize distances of upper diagonal elements to 0
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      D->data[i * n + j] = 0;
    }
  }

  int block_size_1 = 8; // number of samples in block
  int block_size_2 = 8; // number of dimensions in block
  int i = 0;
  for (; i < (n/block_size_1)*block_size_1; i+=block_size_1) {
    int j = i;
    for (; j < (n/block_size_1)*block_size_1; j+=block_size_1) {
      int k = 0;
      for (; k < (m/block_size_2)*block_size_2; k+=block_size_2) {

        for (int ii = i; ii < i+block_size_1; ii++) {
          int jj = (ii+1 > j) ? ii+1 : j;
          for (; jj < j+block_size_1; jj++) {
            double sum = 0;
            for (int kk = k; kk < k+block_size_2; kk++) {
              double dist = X->data[ii * m + kk] - X->data[jj * m + kk];
              sum += dist*dist;
            }
            D->data[ii * n + jj] += sum;
          }
        }
      }

      // remaining dimensions of points
      for (int ii = i; ii < i+block_size_1; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size_1; jj++) {
          double sum = 0;
          for (int kk = k; kk < m; kk++) {
            double dist = X->data[ii * m + kk] - X->data[jj * m + kk];
            sum += dist*dist;
          }
          D->data[ii * n + jj] += sum;
        }
      }
    }

    // remaining columns of D
    for (int ii = i; ii < i+block_size_1; ii++) {
      int jj = (ii+1 > j) ? ii+1 : j;
      for (; jj < n; jj++) {
        double sum = 0;
        for (int kk = 0; kk < m; kk++) {
          double dist = X->data[ii * m + kk] - X->data[jj * m + kk];
          sum += dist*dist;
        }
        D->data[ii * n + jj] += sum;
      }
    }
  }

  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      double sum = 0;
      for (int k = 0; k < m; k++) {
        double dist = X->data[i * m + k] - X->data[j * m + k];
        sum += dist*dist;
      }
      D->data[i * n + j] += sum;
    }
  }


  // set distances of lower diagonal elements
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      D->data[i * n + j] = D->data[j * n + i];
    }
  }
}