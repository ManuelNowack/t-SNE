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