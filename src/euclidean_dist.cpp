#include <float.h>
#include <immintrin.h>
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
        dist = X->data[i * m + k] - X->data[j * m + k];
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
  int end_1 = (n/block_size_1)*block_size_1;
  int end_2 = (m/block_size_2)*block_size_2;
  for (int i = 0; i < end_1; i+=block_size_1) {
    for (int j = i; j < end_1; j+=block_size_1) {
      for (int k = 0; k < end_2; k+=block_size_2) {
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
          for (int kk = end_2; kk < m; kk++) {
            double dist = X->data[ii * m + kk] - X->data[jj * m + kk];
            sum += dist*dist;
          }
          D->data[ii * n + jj] += sum;
        }
      }
    }

    // remaining columns of D
    for (int ii = i; ii < i+block_size_1; ii++) {
      int jj = (ii+1 > end_1) ? ii+1 : end_1;
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
  for (int i = end_1; i < n; i++) {
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

/*
* ALTERNATIVE ALGORITHM
* First compute squared Euclidean norm of every point,
* then use the norms for calculating squared distances.
*/

/*
* Baseline
*/
void euclidean_dist_alt_baseline(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < n; i++) {
    double sum = 0;
    for (int k = 0; k < m; k++) {
      double val = X->data[i * m + k];
      sum += val * val;
    }
    // Store in last row of D.
    norms.data[i] = sum;
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      double scalar_prod = 0;
      for (int k = 0; k < m; k++) {
        double val_i = X->data[i * m + k];
        double val_j = X->data[j * m + k];
        scalar_prod += val_i * val_j;
      }
      double dist = norms.data[i] - 2*scalar_prod + norms.data[j];
      D->data[i * n + j] = dist;
      D->data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Unroll inner-most loops by 2. 
*/
void euclidean_dist_alt_unroll2(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < n; i++) {
    double sum0 = 0;
    double sum1 = 0;
    int k = 0;
    for (; k < 2*(m/2); k+=2) {
      sum0 += X->data[i * m + k] * X->data[i * m + k];
      sum1 += X->data[i * m + k + 1] * X->data[i * m + k + 1];
    }
    for (; k < m; k++) {
      double val = X->data[i * m + k];
      sum0 += val * val;
    }
    // Store in last row of D.
    norms.data[i] = sum0 + sum1;
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      double sum0 = 0;
      double sum1 = 0;
      int k = 0;
      for (; k < 2*(m/2); k+=2) {
        sum0 += X->data[i * m + k] * X->data[j * m + k];
        sum1 += X->data[i * m + k + 1] * X->data[j * m + k + 1];
      }
      for (; k < m; k++) {
        double val_i = X->data[i * m + k];
        double val_j = X->data[j * m + k];
        sum0 += val_i * val_j;
      }
      double dist = norms.data[i] - 2*(sum0 + sum1) + norms.data[j];
      D->data[i * n + j] = dist;
      D->data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Unroll inner-most loops by 4. 
*/
void euclidean_dist_alt_unroll4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < n; i++) {
    double sum0 = 0;
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      sum0 += X->data[i * m + k] * X->data[i * m + k];
      sum1 += X->data[i * m + k + 1] * X->data[i * m + k + 1];
      sum2 += X->data[i * m + k + 2] * X->data[i * m + k + 2];
      sum3 += X->data[i * m + k + 3] * X->data[i * m + k + 3];
    }
    for (; k < m; k++) {
      double val = X->data[i * m + k];
      sum0 += val * val;
    }
    // Store in last row of D.
    norms.data[i] = (sum0 + sum1) + (sum2 + sum3);
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      double sum0 = 0;
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        sum0 += X->data[i * m + k] * X->data[j * m + k];
        sum1 += X->data[i * m + k + 1] * X->data[j * m + k + 1];
        sum2 += X->data[i * m + k + 2] * X->data[j * m + k + 2];
        sum3 += X->data[i * m + k + 3] * X->data[j * m + k + 3];
      }
      for (; k < m; k++) {
        double val_i = X->data[i * m + k];
        double val_j = X->data[j * m + k];
        sum0 += val_i * val_j;
      }
      double dist = norms.data[i] - 2*((sum0 + sum1) + (sum2 + sum3)) + norms.data[j];
      D->data[i * n + j] = dist;
      D->data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Unroll inner-most loops by 8. 
*/
void euclidean_dist_alt_unroll8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < n; i++) {
    double sum0 = 0;
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    double sum4 = 0;
    double sum5 = 0;
    double sum6 = 0;
    double sum7 = 0;
    int k = 0;
    for (; k < 8*(m/8); k+=8) {
      sum0 += X->data[i * m + k] * X->data[i * m + k];
      sum1 += X->data[i * m + k + 1] * X->data[i * m + k + 1];
      sum2 += X->data[i * m + k + 2] * X->data[i * m + k + 2];
      sum3 += X->data[i * m + k + 3] * X->data[i * m + k + 3];
      sum4 += X->data[i * m + k + 4] * X->data[i * m + k + 4];
      sum5 += X->data[i * m + k + 5] * X->data[i * m + k + 5];
      sum6 += X->data[i * m + k + 6] * X->data[i * m + k + 6];
      sum7 += X->data[i * m + k + 7] * X->data[i * m + k + 7];
    }
    for (; k < m; k++) {
      double val = X->data[i * m + k];
      sum0 += val * val;
    }
    // Store in last row of D.
    norms.data[i] = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      double sum0 = 0;
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      double sum4 = 0;
      double sum5 = 0;
      double sum6 = 0;
      double sum7 = 0;
      int k = 0;
      for (; k < 8*(m/8); k+=8) {
        sum0 += X->data[i * m + k] * X->data[j * m + k];
        sum1 += X->data[i * m + k + 1] * X->data[j * m + k + 1];
        sum2 += X->data[i * m + k + 2] * X->data[j * m + k + 2];
        sum3 += X->data[i * m + k + 3] * X->data[j * m + k + 3];
        sum4 += X->data[i * m + k + 4] * X->data[j * m + k + 4];
        sum5 += X->data[i * m + k + 5] * X->data[j * m + k + 5];
        sum6 += X->data[i * m + k + 6] * X->data[j * m + k + 6];
        sum7 += X->data[i * m + k + 7] * X->data[j * m + k + 7];
      }
      for (; k < m; k++) {
        double val_i = X->data[i * m + k];
        double val_j = X->data[j * m + k];
        sum0 += val_i * val_j;
      }
      double sum = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
      double dist = norms.data[i] - 2*sum + norms.data[j];
      D->data[i * n + j] = dist;
      D->data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Unroll inner-most loops by 16. 
*/
void euclidean_dist_alt_unroll16(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < n; i++) {
    double sum0 = 0;
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    double sum4 = 0;
    double sum5 = 0;
    double sum6 = 0;
    double sum7 = 0;
    double sum8 = 0;
    double sum9 = 0;
    double sum10 = 0;
    double sum11 = 0;
    double sum12 = 0;
    double sum13 = 0;
    double sum14 = 0;
    double sum15 = 0;
    int k = 0;
    for (; k < 16*(m/16); k+=16) {
      sum0 += X->data[i * m + k] * X->data[i * m + k];
      sum1 += X->data[i * m + k + 1] * X->data[i * m + k + 1];
      sum2 += X->data[i * m + k + 2] * X->data[i * m + k + 2];
      sum3 += X->data[i * m + k + 3] * X->data[i * m + k + 3];
      sum4 += X->data[i * m + k + 4] * X->data[i * m + k + 4];
      sum5 += X->data[i * m + k + 5] * X->data[i * m + k + 5];
      sum6 += X->data[i * m + k + 6] * X->data[i * m + k + 6];
      sum7 += X->data[i * m + k + 7] * X->data[i * m + k + 7];
      sum8 += X->data[i * m + k + 8] * X->data[i * m + k + 8];
      sum9 += X->data[i * m + k + 9] * X->data[i * m + k + 9];
      sum10 += X->data[i * m + k + 10] * X->data[i * m + k + 10];
      sum11 += X->data[i * m + k + 11] * X->data[i * m + k + 11];
      sum12 += X->data[i * m + k + 12] * X->data[i * m + k + 12];
      sum13 += X->data[i * m + k + 13] * X->data[i * m + k + 13];
      sum14 += X->data[i * m + k + 14] * X->data[i * m + k + 14];
      sum15 += X->data[i * m + k + 15] * X->data[i * m + k + 15];
    }
    for (; k < m; k++) {
      double val = X->data[i * m + k];
      sum0 += val * val;
    }
    // Store in last row of D.
    double total0 = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
    double total1 = ((sum8 + sum9) + (sum10 + sum11)) + ((sum12 + sum13) + (sum14 + sum15));
    norms.data[i] = total0 + total1;
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      double sum0 = 0;
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      double sum4 = 0;
      double sum5 = 0;
      double sum6 = 0;
      double sum7 = 0;
      double sum8 = 0;
      double sum9 = 0;
      double sum10 = 0;
      double sum11 = 0;
      double sum12 = 0;
      double sum13 = 0;
      double sum14 = 0;
      double sum15 = 0;
      int k = 0;
      for (; k < 16*(m/16); k+=16) {
        sum0 += X->data[i * m + k] * X->data[j * m + k];
        sum1 += X->data[i * m + k + 1] * X->data[j * m + k + 1];
        sum2 += X->data[i * m + k + 2] * X->data[j * m + k + 2];
        sum3 += X->data[i * m + k + 3] * X->data[j * m + k + 3];
        sum4 += X->data[i * m + k + 4] * X->data[j * m + k + 4];
        sum5 += X->data[i * m + k + 5] * X->data[j * m + k + 5];
        sum6 += X->data[i * m + k + 6] * X->data[j * m + k + 6];
        sum7 += X->data[i * m + k + 7] * X->data[j * m + k + 7];
        sum8 += X->data[i * m + k + 8] * X->data[j * m + k + 8];
        sum9 += X->data[i * m + k + 9] * X->data[j * m + k + 9];
        sum10 += X->data[i * m + k + 10] * X->data[j * m + k + 10];
        sum11 += X->data[i * m + k + 11] * X->data[j * m + k + 11];
        sum12 += X->data[i * m + k + 12] * X->data[j * m + k + 12];
        sum13 += X->data[i * m + k + 13] * X->data[j * m + k + 13];
        sum14 += X->data[i * m + k + 14] * X->data[j * m + k + 14];
        sum15 += X->data[i * m + k + 15] * X->data[j * m + k + 15];
      }
      for (; k < m; k++) {
        double val_i = X->data[i * m + k];
        double val_j = X->data[j * m + k];
        sum0 += val_i * val_j;
      }
      double total0 = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
      double total1 = ((sum8 + sum9) + (sum10 + sum11)) + ((sum12 + sum13) + (sum14 + sum15));
      double sum = total0 + total1;
      double dist = norms.data[i] - 2*sum + norms.data[j];
      D->data[i * n + j] = dist;
      D->data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Fill Euclidean distance matrix in 4x4 blocks.
*/
void euclidean_dist_alt_block4x4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  const int block_size = 4;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < n; i++) {
    double sum0 = 0;
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    double sum4 = 0;
    double sum5 = 0;
    double sum6 = 0;
    double sum7 = 0;
    int k = 0;
    for (; k < 8*(m/8); k+=8) {
      sum0 += X->data[i * m + k] * X->data[i * m + k];
      sum1 += X->data[i * m + k + 1] * X->data[i * m + k + 1];
      sum2 += X->data[i * m + k + 2] * X->data[i * m + k + 2];
      sum3 += X->data[i * m + k + 3] * X->data[i * m + k + 3];
      sum4 += X->data[i * m + k + 4] * X->data[i * m + k + 4];
      sum5 += X->data[i * m + k + 5] * X->data[i * m + k + 5];
      sum6 += X->data[i * m + k + 6] * X->data[i * m + k + 6];
      sum7 += X->data[i * m + k + 7] * X->data[i * m + k + 7];
    }
    for (; k < m; k++) {
      double val = X->data[i * m + k];
      sum0 += val * val;
    }
    // Store in last row of D.
    norms.data[i] = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
  }

  // Calculate squared Euclidean distances
  int i_block = 0;
  for (; i_block < block_size*(n/block_size); i_block += block_size) {

    int j_block = i_block;
    for (; j_block < block_size*(n/block_size); j_block += block_size) {

      double D_00 = 0;
      double D_01 = 0;
      double D_02 = 0;
      double D_03 = 0;

      double D_10 = 0;
      double D_11 = 0;
      double D_12 = 0;
      double D_13 = 0;

      double D_20 = 0;
      double D_21 = 0;
      double D_22 = 0;
      double D_23 = 0;

      double D_30 = 0;
      double D_31 = 0;
      double D_32 = 0;
      double D_33 = 0;

      for (int k = 0; k < m; k++) {
        D_00 += X->data[i_block * m + k] * X->data[j_block * m + k];
        D_01 += X->data[i_block * m + k] * X->data[j_block * m + 1 * m + k];
        D_02 += X->data[i_block * m + k] * X->data[j_block * m + 2 * m + k];
        D_03 += X->data[i_block * m + k] * X->data[j_block * m + 3 * m + k];

        D_10 += X->data[i_block * m + 1 * m + k] * X->data[j_block * m + k];
        D_11 += X->data[i_block * m + 1 * m + k] * X->data[j_block * m + 1 * m + k];
        D_12 += X->data[i_block * m + 1 * m + k] * X->data[j_block * m + 2 * m + k];
        D_13 += X->data[i_block * m + 1 * m + k] * X->data[j_block * m + 3 * m + k];

        D_20 += X->data[i_block * m + 2 * m + k] * X->data[j_block * m + k];
        D_21 += X->data[i_block * m + 2 * m + k] * X->data[j_block * m + 1 * m + k];
        D_22 += X->data[i_block * m + 2 * m + k] * X->data[j_block * m + 2 * m + k];
        D_23 += X->data[i_block * m + 2 * m + k] * X->data[j_block * m + 3 * m + k];

        D_30 += X->data[i_block * m + 3 * m + k] * X->data[j_block * m + k];
        D_31 += X->data[i_block * m + 3 * m + k] * X->data[j_block * m + 1 * m + k];
        D_32 += X->data[i_block * m + 3 * m + k] * X->data[j_block * m + 2 * m + k];
        D_33 += X->data[i_block * m + 3 * m + k] * X->data[j_block * m + 3 * m + k];
      }

      double norm_i0 = norms.data[i_block];
      double norm_i1 = norms.data[i_block + 1];
      double norm_i2 = norms.data[i_block + 2];
      double norm_i3 = norms.data[i_block + 3];

      double norm_j0 = norms.data[j_block];
      double norm_j1 = norms.data[j_block + 1];
      double norm_j2 = norms.data[j_block + 2];
      double norm_j3 = norms.data[j_block + 3];

      D_00 = norm_i0 + norm_j0 - 2*D_00;
      D_01 = norm_i0 + norm_j1 - 2*D_01;
      D_02 = norm_i0 + norm_j2 - 2*D_02;
      D_03 = norm_i0 + norm_j3 - 2*D_03;

      D_10 = norm_i1 + norm_j0 - 2*D_10;
      D_11 = norm_i1 + norm_j1 - 2*D_11;
      D_12 = norm_i1 + norm_j2 - 2*D_12;
      D_13 = norm_i1 + norm_j3 - 2*D_13;

      D_20 = norm_i2 + norm_j0 - 2*D_20;
      D_21 = norm_i2 + norm_j1 - 2*D_21;
      D_22 = norm_i2 + norm_j2 - 2*D_22;
      D_23 = norm_i2 + norm_j3 - 2*D_23;

      D_30 = norm_i3 + norm_j0 - 2*D_30;
      D_31 = norm_i3 + norm_j1 - 2*D_31;
      D_32 = norm_i3 + norm_j2 - 2*D_32;
      D_33 = norm_i3 + norm_j3 - 2*D_33;

      // Upper triangular elements
      D->data[i_block * n + j_block] = D_00;
      D->data[i_block * n + j_block + 1] = D_01;
      D->data[i_block * n + j_block + 2] = D_02;
      D->data[i_block * n + j_block + 3] = D_03;

      D->data[i_block * n + 1 * n + j_block] = D_10;
      D->data[i_block * n + 1 * n + j_block + 1] = D_11;
      D->data[i_block * n + 1 * n + j_block + 2] = D_12;
      D->data[i_block * n + 1 * n + j_block + 3] = D_13;

      D->data[i_block * n + 2 * n + j_block] = D_20;
      D->data[i_block * n + 2 * n + j_block + 1] = D_21;
      D->data[i_block * n + 2 * n + j_block + 2] = D_22;
      D->data[i_block * n + 2 * n + j_block + 3] = D_23;

      D->data[i_block * n + 3 * n + j_block] = D_30;
      D->data[i_block * n + 3 * n + j_block + 1] = D_31;
      D->data[i_block * n + 3 * n + j_block + 2] = D_32;
      D->data[i_block * n + 3 * n + j_block + 3] = D_33;

      // Lower triangular elements
      D->data[j_block * n + i_block] = D_00;
      D->data[j_block * n + 1 * n + i_block] = D_01;
      D->data[j_block * n + 2 * n + i_block] = D_02;
      D->data[j_block * n + 3 * n + i_block] = D_03;

      D->data[j_block * n + i_block + 1] = D_10;
      D->data[j_block * n + 1 * n + i_block + 1] = D_11;
      D->data[j_block * n + 2 * n + i_block + 1] = D_12;
      D->data[j_block * n + 3 * n + i_block + 1] = D_13;

      D->data[j_block * n + i_block + 2] = D_20;
      D->data[j_block * n + 1 * n + i_block + 2] = D_21;
      D->data[j_block * n + 2 * n + i_block + 2] = D_22;
      D->data[j_block * n + 3 * n + i_block + 2] = D_23;

      D->data[j_block * n + i_block + 3] = D_30;
      D->data[j_block * n + 1 * n + i_block + 3] = D_31;
      D->data[j_block * n + 2 * n + i_block + 3] = D_32;
      D->data[j_block * n + 3 * n + i_block + 3] = D_33;

    }

    // remaining columns
    for (int i = i_block; i < i_block+block_size; i++) {
      for (int j = j_block; j < n; j++) {

        double sum0 = 0;
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;
        double sum4 = 0;
        double sum5 = 0;
        double sum6 = 0;
        double sum7 = 0;
        int k = 0;
        for (; k < 8*(m/8); k+=8) {
          sum0 += X->data[i * m + k] * X->data[j * m + k];
          sum1 += X->data[i * m + k + 1] * X->data[j * m + k + 1];
          sum2 += X->data[i * m + k + 2] * X->data[j * m + k + 2];
          sum3 += X->data[i * m + k + 3] * X->data[j * m + k + 3];
          sum4 += X->data[i * m + k + 4] * X->data[j * m + k + 4];
          sum5 += X->data[i * m + k + 5] * X->data[j * m + k + 5];
          sum6 += X->data[i * m + k + 6] * X->data[j * m + k + 6];
          sum7 += X->data[i * m + k + 7] * X->data[j * m + k + 7];
        }
        for (; k < m; k++) {
          double val_i = X->data[i * m + k];
          double val_j = X->data[j * m + k];
          sum0 += val_i * val_j;
        }
        double sum = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
        double dist = norms.data[i] - 2*sum + norms.data[j];
        D->data[i * n + j] = dist;
        D->data[j * n + i] = dist;

      }
    }
  }

  // remaining bottom right elements
  for (int i = i_block; i < n; i++) {
    for (int j = i+1; j < n; j++) {

      double sum0 = 0;
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      double sum4 = 0;
      double sum5 = 0;
      double sum6 = 0;
      double sum7 = 0;
      int k = 0;
      for (; k < 8*(m/8); k+=8) {
        sum0 += X->data[i * m + k] * X->data[j * m + k];
        sum1 += X->data[i * m + k + 1] * X->data[j * m + k + 1];
        sum2 += X->data[i * m + k + 2] * X->data[j * m + k + 2];
        sum3 += X->data[i * m + k + 3] * X->data[j * m + k + 3];
        sum4 += X->data[i * m + k + 4] * X->data[j * m + k + 4];
        sum5 += X->data[i * m + k + 5] * X->data[j * m + k + 5];
        sum6 += X->data[i * m + k + 6] * X->data[j * m + k + 6];
        sum7 += X->data[i * m + k + 7] * X->data[j * m + k + 7];
      }
      for (; k < m; k++) {
        double val_i = X->data[i * m + k];
        double val_j = X->data[j * m + k];
        sum0 += val_i * val_j;
      }
      double sum = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
      double dist = norms.data[i] - 2*sum + norms.data[j];
      D->data[i * n + j] = dist;
      D->data[j * n + i] = dist;
      
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Vectorize.
*/
void euclidean_dist_alt_vec(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  double *X_data = X->data;
  double *D_data = D->data;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < n; i++) {
    __m256d acc = _mm256_setzero_pd();
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      __m256d x = _mm256_loadu_pd(X_data + i * m + k);
      acc = _mm256_fmadd_pd(x, x, acc);
    }

    // Sum vector
    acc = _mm256_hadd_pd(acc, acc);
    __m256d tmp = _mm256_permute4x64_pd(acc, 0b01001110);
    acc = _mm256_add_pd(acc, tmp);

    // Remaining elements
    double sum = 0;
    for (; k < m; k++) {
      double val = X_data[i * m + k];
      sum += val * val;
    }

    // Store in last row of D.
    norms.data[i] = sum += _mm256_cvtsd_f64(acc);
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      __m256d acc = _mm256_setzero_pd();
      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        __m256d x = _mm256_loadu_pd(X_data + i * m + k);
        __m256d y = _mm256_loadu_pd(X_data + j * m + k);
        acc = _mm256_fmadd_pd(x, y, acc);
      }

      // Sum vector
      acc = _mm256_hadd_pd(acc, acc);
      __m256d tmp = _mm256_permute4x64_pd(acc, 0b01001110);
      acc = _mm256_add_pd(acc, tmp);

      // Remaining elements
      double sum = 0;
      for (; k < m; k++) {
        double val_i = X_data[i * m + k];
        double val_j = X_data[j * m + k];
        sum += val_i * val_j;
      }
      sum += _mm256_cvtsd_f64(acc);

      double dist = norms.data[i] - 2*sum + norms.data[j];
      D_data[i * n + j] = dist;
      D_data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D_data[i * n + i] = 0.0;
  }
}

/*
* Vectorize, unroll by 2.
*/
void euclidean_dist_alt_vec_unroll2(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  double *X_data = X->data;
  double *D_data = D->data;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < 2*(n/2); i+=2) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
      acc0 = _mm256_fmadd_pd(x0, x0, acc0);
      __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
      acc1 = _mm256_fmadd_pd(x1, x1, acc1);
    }

    // Sum vector
    acc0 = _mm256_hadd_pd(acc0, acc0);
    __m256d tmp0 = _mm256_permute4x64_pd(acc0, 0b01001110);
    acc0 = _mm256_add_pd(acc0, tmp0);
    acc1 = _mm256_hadd_pd(acc1, acc1);
    __m256d tmp1 = _mm256_permute4x64_pd(acc1, 0b01001110);
    acc1 = _mm256_add_pd(acc1, tmp1);

    // Remaining elements
    double sum0 = 0;
    double sum1 = 0;
    for (; k < m; k++) {
      double val0 = X_data[i * m + k];
      sum0 += val0 * val0;
      double val1 = X_data[(i+1) * m + k];
      sum1 += val1 * val1;
    }

    // Store in last row of D.
    norms.data[i] = sum0 += _mm256_cvtsd_f64(acc0);
    norms.data[i+1] = sum1 += _mm256_cvtsd_f64(acc1);
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < 2*(n/2); i+=2) {
    for (int j = i+1; j < n; j++) {
      __m256d acc0 = _mm256_setzero_pd();
      __m256d acc1 = _mm256_setzero_pd();
      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
        __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
        __m256d y = _mm256_loadu_pd(X_data + j * m + k);
        acc0 = _mm256_fmadd_pd(x0, y, acc0);
        acc1 = _mm256_fmadd_pd(x1, y, acc1);
      }

      // Sum vector
      acc0 = _mm256_hadd_pd(acc0, acc0);
      __m256d tmp0 = _mm256_permute4x64_pd(acc0, 0b01001110);
      acc0 = _mm256_add_pd(acc0, tmp0);
      acc1 = _mm256_hadd_pd(acc1, acc1);
      __m256d tmp1 = _mm256_permute4x64_pd(acc1, 0b01001110);
      acc1 = _mm256_add_pd(acc1, tmp1);

      // Remaining elements
      double sum0 = 0;
      double sum1 = 0;
      for (; k < m; k++) {
        double val_i_0 = X_data[i * m + k];
        double val_i_1 = X_data[(i+1) * m + k];
        double val_j = X_data[j * m + k];
        sum0 += val_i_0 * val_j;
        sum1 += val_i_1 * val_j;
      }
      sum0 += _mm256_cvtsd_f64(acc0);
      sum1 += _mm256_cvtsd_f64(acc1);

      double dist0 = norms.data[i] - 2*sum0 + norms.data[j];
      double dist1 = norms.data[i+1] - 2*sum1 + norms.data[j];
      D_data[i * n + j] = dist0;
      D_data[j * n + i] = dist0;
      D_data[(i+1) * n + j] = dist1;
      D_data[j * n + i+1] = dist1;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D_data[i * n + i] = 0.0;
  }
}

/*
* Vectorize, unroll by 4.
*/
void euclidean_dist_alt_vec_unroll4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  double *X_data = X->data;
  double *D_data = D->data;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  for (int i = 0; i < 4*(n/4); i+=4) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
      acc0 = _mm256_fmadd_pd(x0, x0, acc0);
      __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
      acc1 = _mm256_fmadd_pd(x1, x1, acc1);
      __m256d x2 = _mm256_loadu_pd(X_data + (i+2) * m + k);
      acc2 = _mm256_fmadd_pd(x2, x2, acc2);
      __m256d x3 = _mm256_loadu_pd(X_data + (i+3) * m + k);
      acc3 = _mm256_fmadd_pd(x3, x3, acc3);
    }

    // Sum vector
    __m256d acc01 = _mm256_hadd_pd(acc0, acc1);
    __m256d tmp01 = _mm256_permute4x64_pd(acc01, 0b01001110);
    acc01 = _mm256_add_pd(acc01, tmp01);

    __m256d acc23 = _mm256_hadd_pd(acc2, acc3);
    __m256d tmp23 = _mm256_permute4x64_pd(acc23, 0b01001110);
    acc23 = _mm256_add_pd(acc23, tmp23);

    // Remaining elements
    double sum0 = 0;
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    for (; k < m; k++) {
      double val0 = X_data[i * m + k];
      sum0 += val0 * val0;
      double val1 = X_data[(i+1) * m + k];
      sum1 += val1 * val1;
      double val2 = X_data[(i+2) * m + k];
      sum2 += val2 * val2;
      double val3 = X_data[(i+3) * m + k];
      sum3 += val3 * val3;
    }

    // Store in last row of D.
    norms.data[i] = sum0 += _mm256_cvtsd_f64(acc01);
    norms.data[i+1] = sum1 += _mm256_cvtsd_f64(_mm256_permute_pd(acc01, 0b0101));
    norms.data[i+2] = sum2 += _mm256_cvtsd_f64(acc23);
    norms.data[i+3] = sum3 += _mm256_cvtsd_f64(_mm256_permute_pd(acc23, 0b0101));
  }

  // Calculate squared Euclidean distances
  for (int i = 0; i < 4*(n/4); i+=4) {
    for (int j = i+1; j < n; j++) {
      __m256d acc0 = _mm256_setzero_pd();
      __m256d acc1 = _mm256_setzero_pd();
      __m256d acc2 = _mm256_setzero_pd();
      __m256d acc3 = _mm256_setzero_pd();
      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
        __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
        __m256d x2 = _mm256_loadu_pd(X_data + (i+2) * m + k);
        __m256d x3 = _mm256_loadu_pd(X_data + (i+3) * m + k);
        __m256d y = _mm256_loadu_pd(X_data + j * m + k);
        acc0 = _mm256_fmadd_pd(x0, y, acc0);
        acc1 = _mm256_fmadd_pd(x1, y, acc1);
        acc2 = _mm256_fmadd_pd(x2, y, acc2);
        acc3 = _mm256_fmadd_pd(x3, y, acc3);
      }

      // Sum vector
      __m256d acc01 = _mm256_hadd_pd(acc0, acc1);
      __m256d tmp01 = _mm256_permute4x64_pd(acc01, 0b01001110);
      acc01 = _mm256_add_pd(acc01, tmp01);

      __m256d acc23 = _mm256_hadd_pd(acc2, acc3);
      __m256d tmp23 = _mm256_permute4x64_pd(acc23, 0b01001110);
      acc23 = _mm256_add_pd(acc23, tmp23);

      // Remaining elements
      double sum0 = 0;
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      for (; k < m; k++) {
        double val_i_0 = X_data[i * m + k];
        double val_i_1 = X_data[(i+1) * m + k];
        double val_i_2 = X_data[(i+2) * m + k];
        double val_i_3 = X_data[(i+3) * m + k];
        double val_j = X_data[j * m + k];
        sum0 += val_i_0 * val_j;
        sum1 += val_i_1 * val_j;
        sum2 += val_i_2 * val_j;
        sum3 += val_i_3 * val_j;
      }
      sum0 += _mm256_cvtsd_f64(acc01);
      sum1 += _mm256_cvtsd_f64(_mm256_permute_pd(acc01, 0b0101));
      sum2 += _mm256_cvtsd_f64(acc23);
      sum3 += _mm256_cvtsd_f64(_mm256_permute_pd(acc23, 0b0101));

      double dist0 = norms.data[i] - 2*sum0 + norms.data[j];
      double dist1 = norms.data[i+1] - 2*sum1 + norms.data[j];
      double dist2 = norms.data[i+2] - 2*sum2 + norms.data[j];
      double dist3 = norms.data[i+3] - 2*sum3 + norms.data[j];
      D_data[i * n + j] = dist0;
      D_data[j * n + i] = dist0;
      D_data[(i+1) * n + j] = dist1;
      D_data[j * n + i+1] = dist1;
      D_data[(i+2) * n + j] = dist2;
      D_data[j * n + i+2] = dist2;
      D_data[(i+3) * n + j] = dist3;
      D_data[j * n + i+3] = dist3;
    }
  }

  // Set diagonal elements
  for (int i = 0; i < n; i++) {
    D_data[i * n + i] = 0.0;
  }
}

/*
* Vectorize, unroll by 8.
*/
void euclidean_dist_alt_vec_unroll8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  double *X_data = X->data;
  double *D_data = D->data;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  int i = 0;
  for (; i < 8*(n/8); i+=8) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    __m256d acc4 = _mm256_setzero_pd();
    __m256d acc5 = _mm256_setzero_pd();
    __m256d acc6 = _mm256_setzero_pd();
    __m256d acc7 = _mm256_setzero_pd();
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
      acc0 = _mm256_fmadd_pd(x0, x0, acc0);
      __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
      acc1 = _mm256_fmadd_pd(x1, x1, acc1);
      __m256d x2 = _mm256_loadu_pd(X_data + (i+2) * m + k);
      acc2 = _mm256_fmadd_pd(x2, x2, acc2);
      __m256d x3 = _mm256_loadu_pd(X_data + (i+3) * m + k);
      acc3 = _mm256_fmadd_pd(x3, x3, acc3);
      __m256d x4 = _mm256_loadu_pd(X_data + (i+4) * m + k);
      acc4 = _mm256_fmadd_pd(x4, x4, acc4);
      __m256d x5 = _mm256_loadu_pd(X_data + (i+5) * m + k);
      acc5 = _mm256_fmadd_pd(x5, x5, acc5);
      __m256d x6 = _mm256_loadu_pd(X_data + (i+6) * m + k);
      acc6 = _mm256_fmadd_pd(x6, x6, acc6);
      __m256d x7 = _mm256_loadu_pd(X_data + (i+7) * m + k);
      acc7 = _mm256_fmadd_pd(x7, x7, acc7);
    }

    // Sum vector
    __m256d acc01 = _mm256_hadd_pd(acc0, acc1);
    __m256d tmp01 = _mm256_permute4x64_pd(acc01, 0b01001110);
    acc01 = _mm256_add_pd(acc01, tmp01);

    __m256d acc23 = _mm256_hadd_pd(acc2, acc3);
    __m256d tmp23 = _mm256_permute4x64_pd(acc23, 0b01001110);
    acc23 = _mm256_add_pd(acc23, tmp23);

    __m256d acc45 = _mm256_hadd_pd(acc4, acc5);
    __m256d tmp45 = _mm256_permute4x64_pd(acc45, 0b01001110);
    acc45 = _mm256_add_pd(acc45, tmp45);

    __m256d acc67 = _mm256_hadd_pd(acc6, acc7);
    __m256d tmp67 = _mm256_permute4x64_pd(acc67, 0b01001110);
    acc67 = _mm256_add_pd(acc67, tmp67);

  
    // Remaining elements
    double sum0 = 0;
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    double sum4 = 0;
    double sum5 = 0;
    double sum6 = 0;
    double sum7 = 0;
    for (; k < m; k++) {
      double val0 = X_data[i * m + k];
      sum0 += val0 * val0;
      double val1 = X_data[(i+1) * m + k];
      sum1 += val1 * val1;
      double val2 = X_data[(i+2) * m + k];
      sum2 += val2 * val2;
      double val3 = X_data[(i+3) * m + k];
      sum3 += val3 * val3;
      double val4 = X_data[(i+4) * m + k];
      sum4 += val4 * val4;
      double val5 = X_data[(i+5) * m + k];
      sum5 += val5 * val5;
      double val6 = X_data[(i+6) * m + k];
      sum6 += val6 * val6;
      double val7 = X_data[(i+7) * m + k];
      sum7 += val7 * val7;
    }

    // Store in last row of D.
    norms.data[i] = sum0 += _mm256_cvtsd_f64(acc01);
    norms.data[i+1] = sum1 += _mm256_cvtsd_f64(_mm256_permute_pd(acc01, 0b0101));

    norms.data[i+2] = sum2 += _mm256_cvtsd_f64(acc23);
    norms.data[i+3] = sum3 += _mm256_cvtsd_f64(_mm256_permute_pd(acc23, 0b0101));

    norms.data[i+4] = sum4 += _mm256_cvtsd_f64(acc45);
    norms.data[i+5] = sum5 += _mm256_cvtsd_f64(_mm256_permute_pd(acc45, 0b0101));

    norms.data[i+6] = sum6 += _mm256_cvtsd_f64(acc67);
    norms.data[i+7] = sum7 += _mm256_cvtsd_f64(_mm256_permute_pd(acc67, 0b0101));
  }
  for (; i < n; i++) {
    __m256d acc = _mm256_setzero_pd();
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      __m256d x = _mm256_loadu_pd(X_data + i * m + k);
      acc = _mm256_fmadd_pd(x, x, acc);
    }

    // Sum vector
    acc = _mm256_hadd_pd(acc, acc);
    __m256d tmp = _mm256_permute4x64_pd(acc, 0b01001110);
    acc = _mm256_add_pd(acc, tmp);

    // Remaining elements
    double sum = 0;
    for (; k < m; k++) {
      double val = X_data[i * m + k];
      sum += val * val;
    }

    // Store in last row of D.
    norms.data[i] = sum += _mm256_cvtsd_f64(acc);
  }

  // Calculate squared Euclidean distances
  i = 0;
  for (; i < 8*(n/8); i+=8) {
    for (int j = i+1; j < n; j++) {
      __m256d acc0 = _mm256_setzero_pd();
      __m256d acc1 = _mm256_setzero_pd();
      __m256d acc2 = _mm256_setzero_pd();
      __m256d acc3 = _mm256_setzero_pd();
      __m256d acc4 = _mm256_setzero_pd();
      __m256d acc5 = _mm256_setzero_pd();
      __m256d acc6 = _mm256_setzero_pd();
      __m256d acc7 = _mm256_setzero_pd();
      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
        __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
        __m256d x2 = _mm256_loadu_pd(X_data + (i+2) * m + k);
        __m256d x3 = _mm256_loadu_pd(X_data + (i+3) * m + k);
        __m256d x4 = _mm256_loadu_pd(X_data + (i+4) * m + k);
        __m256d x5 = _mm256_loadu_pd(X_data + (i+5) * m + k);
        __m256d x6 = _mm256_loadu_pd(X_data + (i+6) * m + k);
        __m256d x7 = _mm256_loadu_pd(X_data + (i+7) * m + k);
        __m256d y = _mm256_loadu_pd(X_data + j * m + k);
        acc0 = _mm256_fmadd_pd(x0, y, acc0);
        acc1 = _mm256_fmadd_pd(x1, y, acc1);
        acc2 = _mm256_fmadd_pd(x2, y, acc2);
        acc3 = _mm256_fmadd_pd(x3, y, acc3);
        acc4 = _mm256_fmadd_pd(x4, y, acc4);
        acc5 = _mm256_fmadd_pd(x5, y, acc5);
        acc6 = _mm256_fmadd_pd(x6, y, acc6);
        acc7 = _mm256_fmadd_pd(x7, y, acc7);
      }

      // Sum vector
      __m256d acc01 = _mm256_hadd_pd(acc0, acc1);
      __m256d tmp01 = _mm256_permute4x64_pd(acc01, 0b01001110);
      acc01 = _mm256_add_pd(acc01, tmp01);

      __m256d acc23 = _mm256_hadd_pd(acc2, acc3);
      __m256d tmp23 = _mm256_permute4x64_pd(acc23, 0b01001110);
      acc23 = _mm256_add_pd(acc23, tmp23);

      __m256d acc45 = _mm256_hadd_pd(acc4, acc5);
      __m256d tmp45 = _mm256_permute4x64_pd(acc45, 0b01001110);
      acc45 = _mm256_add_pd(acc45, tmp45);

      __m256d acc67 = _mm256_hadd_pd(acc6, acc7);
      __m256d tmp67 = _mm256_permute4x64_pd(acc67, 0b01001110);
      acc67 = _mm256_add_pd(acc67, tmp67);


      // Remaining elements
      double sum0 = 0;
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      double sum4 = 0;
      double sum5 = 0;
      double sum6 = 0;
      double sum7 = 0;
      for (; k < m; k++) {
        double val_i_0 = X_data[i * m + k];
        double val_i_1 = X_data[(i+1) * m + k];
        double val_i_2 = X_data[(i+2) * m + k];
        double val_i_3 = X_data[(i+3) * m + k];
        double val_i_4 = X_data[(i+4) * m + k];
        double val_i_5 = X_data[(i+5) * m + k];
        double val_i_6 = X_data[(i+6) * m + k];
        double val_i_7 = X_data[(i+7) * m + k];
        double val_j = X_data[j * m + k];
        sum0 += val_i_0 * val_j;
        sum1 += val_i_1 * val_j;
        sum2 += val_i_2 * val_j;
        sum3 += val_i_3 * val_j;
        sum4 += val_i_4 * val_j;
        sum5 += val_i_5 * val_j;
        sum6 += val_i_6 * val_j;
        sum7 += val_i_7 * val_j;
      }
      sum0 += _mm256_cvtsd_f64(acc01);
      sum1 += _mm256_cvtsd_f64(_mm256_permute_pd(acc01, 0b0101));
      sum2 += _mm256_cvtsd_f64(acc23);
      sum3 += _mm256_cvtsd_f64(_mm256_permute_pd(acc23, 0b0101));
      sum4 += _mm256_cvtsd_f64(acc45);
      sum5 += _mm256_cvtsd_f64(_mm256_permute_pd(acc45, 0b0101));
      sum6 += _mm256_cvtsd_f64(acc67);
      sum7 += _mm256_cvtsd_f64(_mm256_permute_pd(acc67, 0b0101));

      double dist0 = norms.data[i] - 2*sum0 + norms.data[j];
      double dist1 = norms.data[i+1] - 2*sum1 + norms.data[j];
      double dist2 = norms.data[i+2] - 2*sum2 + norms.data[j];
      double dist3 = norms.data[i+3] - 2*sum3 + norms.data[j];
      double dist4 = norms.data[i+4] - 2*sum4 + norms.data[j];
      double dist5 = norms.data[i+5] - 2*sum5 + norms.data[j];
      double dist6 = norms.data[i+6] - 2*sum6 + norms.data[j];
      double dist7 = norms.data[i+7] - 2*sum7 + norms.data[j];
      
      D_data[i * n + j] = dist0;
      D_data[(i+1) * n + j] = dist1;
      D_data[(i+2) * n + j] = dist2;
      D_data[(i+3) * n + j] = dist3;
      D_data[(i+4) * n + j] = dist4;
      D_data[(i+5) * n + j] = dist5;
      D_data[(i+6) * n + j] = dist6;
      D_data[(i+7) * n + j] = dist7;

      D_data[j * n + i] = dist0;
      D_data[j * n + i+1] = dist1;
      D_data[j * n + i+2] = dist2;
      D_data[j * n + i+3] = dist3;

      D_data[j * n + i+4] = dist4;
      D_data[j * n + i+5] = dist5;
      D_data[j * n + i+6] = dist6;
      D_data[j * n + i+7] = dist7;
    }
  }
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      __m256d acc = _mm256_setzero_pd();
      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        __m256d x = _mm256_loadu_pd(X_data + i * m + k);
        __m256d y = _mm256_loadu_pd(X_data + j * m + k);
        acc = _mm256_fmadd_pd(x, y, acc);
      }

      // Sum vector
      acc = _mm256_hadd_pd(acc, acc);
      __m256d tmp = _mm256_permute4x64_pd(acc, 0b01001110);
      acc = _mm256_add_pd(acc, tmp);

      // Remaining elements
      double sum = 0;
      for (; k < m; k++) {
        double val_i = X_data[i * m + k];
        double val_j = X_data[j * m + k];
        sum += val_i * val_j;
      }
      sum += _mm256_cvtsd_f64(acc);

      double dist = norms.data[i] - 2*sum + norms.data[j];
      D_data[i * n + j] = dist;
      D_data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int j = 0; j < n; j++) {
    D_data[j * n + j] = 0.0;
  }
}

/*
* Vectorize, unroll by 4x4.
*/
void euclidean_dist_alt_vec_unroll4x4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  double *X_data = X->data;
  double *D_data = D->data;

  // Pointer to last row of D
  Matrix norms = create_matrix(n, 1);

  // Pre-compute squared Euclidean norms
  int i = 0;
  for (; i < 8*(n/8); i+=8) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    __m256d acc4 = _mm256_setzero_pd();
    __m256d acc5 = _mm256_setzero_pd();
    __m256d acc6 = _mm256_setzero_pd();
    __m256d acc7 = _mm256_setzero_pd();
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
      acc0 = _mm256_fmadd_pd(x0, x0, acc0);
      __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
      acc1 = _mm256_fmadd_pd(x1, x1, acc1);
      __m256d x2 = _mm256_loadu_pd(X_data + (i+2) * m + k);
      acc2 = _mm256_fmadd_pd(x2, x2, acc2);
      __m256d x3 = _mm256_loadu_pd(X_data + (i+3) * m + k);
      acc3 = _mm256_fmadd_pd(x3, x3, acc3);
      __m256d x4 = _mm256_loadu_pd(X_data + (i+4) * m + k);
      acc4 = _mm256_fmadd_pd(x4, x4, acc4);
      __m256d x5 = _mm256_loadu_pd(X_data + (i+5) * m + k);
      acc5 = _mm256_fmadd_pd(x5, x5, acc5);
      __m256d x6 = _mm256_loadu_pd(X_data + (i+6) * m + k);
      acc6 = _mm256_fmadd_pd(x6, x6, acc6);
      __m256d x7 = _mm256_loadu_pd(X_data + (i+7) * m + k);
      acc7 = _mm256_fmadd_pd(x7, x7, acc7);
    }

    // Sum vector
    __m256d acc01 = _mm256_hadd_pd(acc0, acc1);
    __m256d tmp01 = _mm256_permute4x64_pd(acc01, 0b01001110);
    acc01 = _mm256_add_pd(acc01, tmp01);

    __m256d acc23 = _mm256_hadd_pd(acc2, acc3);
    __m256d tmp23 = _mm256_permute4x64_pd(acc23, 0b01001110);
    acc23 = _mm256_add_pd(acc23, tmp23);

    __m256d acc45 = _mm256_hadd_pd(acc4, acc5);
    __m256d tmp45 = _mm256_permute4x64_pd(acc45, 0b01001110);
    acc45 = _mm256_add_pd(acc45, tmp45);

    __m256d acc67 = _mm256_hadd_pd(acc6, acc7);
    __m256d tmp67 = _mm256_permute4x64_pd(acc67, 0b01001110);
    acc67 = _mm256_add_pd(acc67, tmp67);

  
    // Remaining elements
    double sum0 = 0;
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    double sum4 = 0;
    double sum5 = 0;
    double sum6 = 0;
    double sum7 = 0;
    for (; k < m; k++) {
      double val0 = X_data[i * m + k];
      sum0 += val0 * val0;
      double val1 = X_data[(i+1) * m + k];
      sum1 += val1 * val1;
      double val2 = X_data[(i+2) * m + k];
      sum2 += val2 * val2;
      double val3 = X_data[(i+3) * m + k];
      sum3 += val3 * val3;
      double val4 = X_data[(i+4) * m + k];
      sum4 += val4 * val4;
      double val5 = X_data[(i+5) * m + k];
      sum5 += val5 * val5;
      double val6 = X_data[(i+6) * m + k];
      sum6 += val6 * val6;
      double val7 = X_data[(i+7) * m + k];
      sum7 += val7 * val7;
    }

    // Store in last row of D.
    norms.data[i] = sum0 += _mm256_cvtsd_f64(acc01);
    norms.data[i+1] = sum1 += _mm256_cvtsd_f64(_mm256_permute_pd(acc01, 0b0101));

    norms.data[i+2] = sum2 += _mm256_cvtsd_f64(acc23);
    norms.data[i+3] = sum3 += _mm256_cvtsd_f64(_mm256_permute_pd(acc23, 0b0101));

    norms.data[i+4] = sum4 += _mm256_cvtsd_f64(acc45);
    norms.data[i+5] = sum5 += _mm256_cvtsd_f64(_mm256_permute_pd(acc45, 0b0101));

    norms.data[i+6] = sum6 += _mm256_cvtsd_f64(acc67);
    norms.data[i+7] = sum7 += _mm256_cvtsd_f64(_mm256_permute_pd(acc67, 0b0101));
  }
  for (; i < n; i++) {
    __m256d acc = _mm256_setzero_pd();
    int k = 0;
    for (; k < 4*(m/4); k+=4) {
      __m256d x = _mm256_loadu_pd(X_data + i * m + k);
      acc = _mm256_fmadd_pd(x, x, acc);
    }

    // Sum vector
    acc = _mm256_hadd_pd(acc, acc);
    __m256d tmp = _mm256_permute4x64_pd(acc, 0b01001110);
    acc = _mm256_add_pd(acc, tmp);

    // Remaining elements
    double sum = 0;
    for (; k < m; k++) {
      double val = X_data[i * m + k];
      sum += val * val;
    }

    // Store in last row of D.
    norms.data[i] = sum += _mm256_cvtsd_f64(acc);
  }

  // Calculate squared Euclidean distances
  i = 0;
  for (; i < 4*(n/4); i+=4) {
    for (int j = i; j < 4*(n/4); j+=4) {
      __m256d acc00 = _mm256_setzero_pd();
      __m256d acc01 = _mm256_setzero_pd();
      __m256d acc02 = _mm256_setzero_pd();
      __m256d acc03 = _mm256_setzero_pd();

      __m256d acc10 = _mm256_setzero_pd();
      __m256d acc11 = _mm256_setzero_pd();
      __m256d acc12 = _mm256_setzero_pd();
      __m256d acc13 = _mm256_setzero_pd();

      __m256d acc20 = _mm256_setzero_pd();
      __m256d acc21 = _mm256_setzero_pd();
      __m256d acc22 = _mm256_setzero_pd();
      __m256d acc23 = _mm256_setzero_pd();

      __m256d acc30 = _mm256_setzero_pd();
      __m256d acc31 = _mm256_setzero_pd();
      __m256d acc32 = _mm256_setzero_pd();
      __m256d acc33 = _mm256_setzero_pd();

      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        __m256d x0 = _mm256_loadu_pd(X_data + i * m + k);
        __m256d x1 = _mm256_loadu_pd(X_data + (i+1) * m + k);
        __m256d x2 = _mm256_loadu_pd(X_data + (i+2) * m + k);
        __m256d x3 = _mm256_loadu_pd(X_data + (i+3) * m + k);

        __m256d y0 = _mm256_loadu_pd(X_data + j * m + k);
        __m256d y1 = _mm256_loadu_pd(X_data + (j+1) * m + k);
        __m256d y2 = _mm256_loadu_pd(X_data + (j+2) * m + k);
        __m256d y3 = _mm256_loadu_pd(X_data + (j+3) * m + k);

        acc00 = _mm256_fmadd_pd(x0, y0, acc00);
        acc01 = _mm256_fmadd_pd(x0, y1, acc01);
        acc02 = _mm256_fmadd_pd(x0, y2, acc02);
        acc03 = _mm256_fmadd_pd(x0, y3, acc03);

        acc10 = _mm256_fmadd_pd(x1, y0, acc10);
        acc11 = _mm256_fmadd_pd(x1, y1, acc11);
        acc12 = _mm256_fmadd_pd(x1, y2, acc12);
        acc13 = _mm256_fmadd_pd(x1, y3, acc13);

        acc20 = _mm256_fmadd_pd(x2, y0, acc20);
        acc21 = _mm256_fmadd_pd(x2, y1, acc21);
        acc22 = _mm256_fmadd_pd(x2, y2, acc22);
        acc23 = _mm256_fmadd_pd(x2, y3, acc23);

        acc30 = _mm256_fmadd_pd(x3, y0, acc30);
        acc31 = _mm256_fmadd_pd(x3, y1, acc31);
        acc32 = _mm256_fmadd_pd(x3, y2, acc32);
        acc33 = _mm256_fmadd_pd(x3, y3, acc33);
      }

      // Sum vector
      __m256d acc0001 = _mm256_hadd_pd(acc00, acc01);
      __m256d tmp0001 = _mm256_permute4x64_pd(acc0001, 0b01001110);
      acc0001 = _mm256_add_pd(acc0001, tmp0001);
      __m256d acc0203 = _mm256_hadd_pd(acc02, acc03);
      __m256d tmp0203 = _mm256_permute4x64_pd(acc0203, 0b01001110);
      acc0203 = _mm256_add_pd(acc0203, tmp0203);

      __m256d acc1011 = _mm256_hadd_pd(acc10, acc11);
      __m256d tmp1011 = _mm256_permute4x64_pd(acc1011, 0b01001110);
      acc1011 = _mm256_add_pd(acc1011, tmp1011);
      __m256d acc1213 = _mm256_hadd_pd(acc12, acc13);
      __m256d tmp1213 = _mm256_permute4x64_pd(acc1213, 0b01001110);
      acc1213 = _mm256_add_pd(acc1213, tmp1213);

      __m256d acc2021 = _mm256_hadd_pd(acc20, acc21);
      __m256d tmp2021 = _mm256_permute4x64_pd(acc2021, 0b01001110);
      acc2021 = _mm256_add_pd(acc2021, tmp2021);
      __m256d acc2223 = _mm256_hadd_pd(acc22, acc23);
      __m256d tmp2223 = _mm256_permute4x64_pd(acc2223, 0b01001110);
      acc2223 = _mm256_add_pd(acc2223, tmp2223);

      __m256d acc3031 = _mm256_hadd_pd(acc30, acc31);
      __m256d tmp3031 = _mm256_permute4x64_pd(acc3031, 0b01001110);
      acc3031 = _mm256_add_pd(acc3031, tmp3031);
      __m256d acc3233 = _mm256_hadd_pd(acc32, acc33);
      __m256d tmp3233 = _mm256_permute4x64_pd(acc3233, 0b01001110);
      acc3233 = _mm256_add_pd(acc3233, tmp3233);


      // Remaining elements
      double sum00 = 0;
      double sum01 = 0;
      double sum02 = 0;
      double sum03 = 0;

      double sum10 = 0;
      double sum11 = 0;
      double sum12 = 0;
      double sum13 = 0;

      double sum20 = 0;
      double sum21 = 0;
      double sum22 = 0;
      double sum23 = 0;

      double sum30 = 0;
      double sum31 = 0;
      double sum32 = 0;
      double sum33 = 0;

      for (; k < m; k++) {
        double val_i_0 = X_data[i * m + k];
        double val_i_1 = X_data[(i+1) * m + k];
        double val_i_2 = X_data[(i+2) * m + k];
        double val_i_3 = X_data[(i+3) * m + k];
        
        double val_j_0 = X_data[j * m + k];
        double val_j_1 = X_data[(j+1) * m + k];
        double val_j_2 = X_data[(j+2) * m + k];
        double val_j_3 = X_data[(j+3) * m + k];

        sum00 += val_i_0 * val_j_0;
        sum01 += val_i_0 * val_j_1;
        sum02 += val_i_0 * val_j_2;
        sum03 += val_i_0 * val_j_3;

        sum10 += val_i_1 * val_j_0;
        sum11 += val_i_1 * val_j_1;
        sum12 += val_i_1 * val_j_2;
        sum13 += val_i_1 * val_j_3;

        sum20 += val_i_2 * val_j_0;
        sum21 += val_i_2 * val_j_1;
        sum22 += val_i_2 * val_j_2;
        sum23 += val_i_2 * val_j_3;

        sum30 += val_i_3 * val_j_0;
        sum31 += val_i_3 * val_j_1;
        sum32 += val_i_3 * val_j_2;
        sum33 += val_i_3 * val_j_3;
      }

      sum00 += _mm256_cvtsd_f64(acc0001);
      sum01 += _mm256_cvtsd_f64(_mm256_permute_pd(acc0001, 0b0101));
      sum02 += _mm256_cvtsd_f64(acc0203);
      sum03 += _mm256_cvtsd_f64(_mm256_permute_pd(acc0203, 0b0101));

      sum10 += _mm256_cvtsd_f64(acc1011);
      sum11 += _mm256_cvtsd_f64(_mm256_permute_pd(acc1011, 0b0101));
      sum12 += _mm256_cvtsd_f64(acc1213);
      sum13 += _mm256_cvtsd_f64(_mm256_permute_pd(acc1213, 0b0101));

      sum20 += _mm256_cvtsd_f64(acc2021);
      sum21 += _mm256_cvtsd_f64(_mm256_permute_pd(acc2021, 0b0101));
      sum22 += _mm256_cvtsd_f64(acc2223);
      sum23 += _mm256_cvtsd_f64(_mm256_permute_pd(acc2223, 0b0101));

      sum30 += _mm256_cvtsd_f64(acc3031);
      sum31 += _mm256_cvtsd_f64(_mm256_permute_pd(acc3031, 0b0101));
      sum32 += _mm256_cvtsd_f64(acc3233);
      sum33 += _mm256_cvtsd_f64(_mm256_permute_pd(acc3233, 0b0101));

      double dist00 = norms.data[i] - 2*sum00 + norms.data[j];
      double dist01 = norms.data[i] - 2*sum01 + norms.data[j+1];
      double dist02 = norms.data[i] - 2*sum02 + norms.data[j+2];
      double dist03 = norms.data[i] - 2*sum03 + norms.data[j+3];

      double dist10 = norms.data[i+1] - 2*sum10 + norms.data[j];
      double dist11 = norms.data[i+1] - 2*sum11 + norms.data[j+1];
      double dist12 = norms.data[i+1] - 2*sum12 + norms.data[j+2];
      double dist13 = norms.data[i+1] - 2*sum13 + norms.data[j+3];

      double dist20 = norms.data[i+2] - 2*sum20 + norms.data[j];
      double dist21 = norms.data[i+2] - 2*sum21 + norms.data[j+1];
      double dist22 = norms.data[i+2] - 2*sum22 + norms.data[j+2];
      double dist23 = norms.data[i+2] - 2*sum23 + norms.data[j+3];

      double dist30 = norms.data[i+3] - 2*sum30 + norms.data[j];
      double dist31 = norms.data[i+3] - 2*sum31 + norms.data[j+1];
      double dist32 = norms.data[i+3] - 2*sum32 + norms.data[j+2];
      double dist33 = norms.data[i+3] - 2*sum33 + norms.data[j+3];

      D_data[i * n + j] = dist00;
      D_data[i * n + j + 1] = dist01;
      D_data[i * n + j + 2] = dist02;
      D_data[i * n + j + 3] = dist03;

      D_data[(i+1) * n + j] = dist10;
      D_data[(i+1) * n + j + 1] = dist11;
      D_data[(i+1) * n + j + 2] = dist12;
      D_data[(i+1) * n + j + 3] = dist13;

      D_data[(i+2) * n + j] = dist20;
      D_data[(i+2) * n + j + 1] = dist21;
      D_data[(i+2) * n + j + 2] = dist22;
      D_data[(i+2) * n + j + 3] = dist23;

      D_data[(i+3) * n + j] = dist30;
      D_data[(i+3) * n + j + 1] = dist31;
      D_data[(i+3) * n + j + 2] = dist32;
      D_data[(i+3) * n + j + 3] = dist33;


      D_data[j * n + i] = dist00;
      D_data[j * n + i + 1] = dist10;
      D_data[j * n + i + 2] = dist20;
      D_data[j * n + i + 3] = dist30;

      D_data[(j+1) * n + i] = dist01;
      D_data[(j+1) * n + i + 1] = dist11;
      D_data[(j+1) * n + i + 2] = dist21;
      D_data[(j+1) * n + i + 3] = dist31;

      D_data[(j+2) * n + i] = dist02;
      D_data[(j+2) * n + i + 1] = dist12;
      D_data[(j+2) * n + i + 2] = dist22;
      D_data[(j+2) * n + i + 3] = dist32;

      D_data[(j+3) * n + i] = dist03;
      D_data[(j+3) * n + i + 1] = dist13;
      D_data[(j+3) * n + i + 2] = dist23;
      D_data[(j+3) * n + i + 3] = dist33;
    }
  }
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      __m256d acc = _mm256_setzero_pd();
      int k = 0;
      for (; k < 4*(m/4); k+=4) {
        __m256d x = _mm256_loadu_pd(X_data + i * m + k);
        __m256d y = _mm256_loadu_pd(X_data + j * m + k);
        acc = _mm256_fmadd_pd(x, y, acc);
      }

      // Sum vector
      acc = _mm256_hadd_pd(acc, acc);
      __m256d tmp = _mm256_permute4x64_pd(acc, 0b01001110);
      acc = _mm256_add_pd(acc, tmp);

      // Remaining elements
      double sum = 0;
      for (; k < m; k++) {
        double val_i = X_data[i * m + k];
        double val_j = X_data[j * m + k];
        sum += val_i * val_j;
      }
      sum += _mm256_cvtsd_f64(acc);

      double dist = norms.data[i] - 2*sum + norms.data[j];
      D_data[i * n + j] = dist;
      D_data[j * n + i] = dist;
    }
  }

  // Set diagonal elements
  for (int j = 0; j < n; j++) {
    D_data[j * n + j] = 0.0;
  }
}
