#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/debug.h>
#include <tsne/matrix.h>

/*
* Squared Euclidean distance functions optimised for two-dimensional points
*/

/*
* As baseline, but calculate upper triangular elements only
*/
void euclidean_dist_low_upper(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // calculate non-diagonal entries
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Euclidean distance
      double sum = 0;
      for (int k = 0; k < m; k++) {
        double dist = X->data[i * m + k] - X->data[j * m + k];
        sum += dist * dist;
      }
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Unroll iteration over point dimensions.
*/
void euclidean_dist_low_unroll(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // calculate non-diagonal entries
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Calculate distance in blocks of 2 points.
*/
void euclidean_dist_low_block2(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  int block_size = 2;

  // calculate non-diagonal entries
  int i = 0;
  for (; i < block_size*(n/block_size); i+=block_size) {
    int j = i;
    for (; j < block_size*(n/block_size); j+=block_size) {
      // full block
      for (int ii = i; ii < i+block_size; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size; jj++) {
          // Euclidean distance
          double dist0 = X->data[ii * m] - X->data[jj * m];
          double dist1 = X->data[ii * m + 1] - X->data[jj * m + 1];
          double sum = dist0*dist0 + dist1*dist1;
          D->data[ii * n + jj] = sum;
        }
      }
    }
    // remaining columns
    for (; j < n; j++) {
      for (int ii = i; ii < i+block_size; ii++) {
        // Euclidean distance
        double dist0 = X->data[ii * m] - X->data[j * m];
        double dist1 = X->data[ii * m + 1] - X->data[j * m + 1];
        double sum = dist0*dist0 + dist1*dist1;
        D->data[ii * n + j] = sum;
      }
    }
  }
  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Calculate distance in blocks of 4 points.
*/
void euclidean_dist_low_block4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  int block_size = 4;

  // calculate non-diagonal entries
  int i = 0;
  for (; i < block_size*(n/block_size); i+=block_size) {
    int j = i;
    for (; j < block_size*(n/block_size); j+=block_size) {
      // full block
      for (int ii = i; ii < i+block_size; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size; jj++) {
          // Euclidean distance
          double dist0 = X->data[ii * m] - X->data[jj * m];
          double dist1 = X->data[ii * m + 1] - X->data[jj * m + 1];
          double sum = dist0*dist0 + dist1*dist1;
          D->data[ii * n + jj] = sum;
        }
      }
    }
    // remaining columns
    for (; j < n; j++) {
      for (int ii = i; ii < i+block_size; ii++) {
        // Euclidean distance
        double dist0 = X->data[ii * m] - X->data[j * m];
        double dist1 = X->data[ii * m + 1] - X->data[j * m + 1];
        double sum = dist0*dist0 + dist1*dist1;
        D->data[ii * n + j] = sum;
      }
    }
  }
  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Calculate distance in blocks of 8 points.
*/
void euclidean_dist_low_block8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  int block_size = 8;

  // calculate non-diagonal entries
  int i = 0;
  for (; i < block_size*(n/block_size); i+=block_size) {
    int j = i;
    for (; j < block_size*(n/block_size); j+=block_size) {
      // full block
      for (int ii = i; ii < i+block_size; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size; jj++) {
          // Euclidean distance
          double dist0 = X->data[ii * m] - X->data[jj * m];
          double dist1 = X->data[ii * m + 1] - X->data[jj * m + 1];
          double sum = dist0*dist0 + dist1*dist1;
          D->data[ii * n + jj] = sum;
        }
      }
    }
    // remaining columns
    for (; j < n; j++) {
      for (int ii = i; ii < i+block_size; ii++) {
        // Euclidean distance
        double dist0 = X->data[ii * m] - X->data[j * m];
        double dist1 = X->data[ii * m + 1] - X->data[j * m + 1];
        double sum = dist0*dist0 + dist1*dist1;
        D->data[ii * n + j] = sum;
      }
    }
  }
  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Calculate distance in blocks of 16 points.
*/
void euclidean_dist_low_block16(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  int block_size = 16;

  // calculate non-diagonal entries
  int i = 0;
  for (; i < block_size*(n/block_size); i+=block_size) {
    int j = i;
    for (; j < block_size*(n/block_size); j+=block_size) {
      // full block
      for (int ii = i; ii < i+block_size; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size; jj++) {
          // Euclidean distance
          double dist0 = X->data[ii * m] - X->data[jj * m];
          double dist1 = X->data[ii * m + 1] - X->data[jj * m + 1];
          double sum = dist0*dist0 + dist1*dist1;
          D->data[ii * n + jj] = sum;
        }
      }
    }
    // remaining columns
    for (; j < n; j++) {
      for (int ii = i; ii < i+block_size; ii++) {
        // Euclidean distance
        double dist0 = X->data[ii * m] - X->data[j * m];
        double dist1 = X->data[ii * m + 1] - X->data[j * m + 1];
        double sum = dist0*dist0 + dist1*dist1;
        D->data[ii * n + j] = sum;
      }
    }
  }
  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Calculate distance in blocks of 32 points.
*/
void euclidean_dist_low_block32(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  int block_size = 32;

  // calculate non-diagonal entries
  int i = 0;
  for (; i < block_size*(n/block_size); i+=block_size) {
    int j = i;
    for (; j < block_size*(n/block_size); j+=block_size) {
      // full block
      for (int ii = i; ii < i+block_size; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size; jj++) {
          // Euclidean distance
          double dist0 = X->data[ii * m] - X->data[jj * m];
          double dist1 = X->data[ii * m + 1] - X->data[jj * m + 1];
          double sum = dist0*dist0 + dist1*dist1;
          D->data[ii * n + jj] = sum;
        }
      }
    }
    // remaining columns
    for (; j < n; j++) {
      for (int ii = i; ii < i+block_size; ii++) {
        // Euclidean distance
        double dist0 = X->data[ii * m] - X->data[j * m];
        double dist1 = X->data[ii * m + 1] - X->data[j * m + 1];
        double sum = dist0*dist0 + dist1*dist1;
        D->data[ii * n + j] = sum;
      }
    }
  }
  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Calculate distance in blocks of 64 points.
*/
void euclidean_dist_low_block64(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  int block_size = 64;

  // calculate non-diagonal entries
  int i = 0;
  for (; i < block_size*(n/block_size); i+=block_size) {
    int j = i;
    for (; j < block_size*(n/block_size); j+=block_size) {
      // full block
      for (int ii = i; ii < i+block_size; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size; jj++) {
          // Euclidean distance
          double dist0 = X->data[ii * m] - X->data[jj * m];
          double dist1 = X->data[ii * m + 1] - X->data[jj * m + 1];
          double sum = dist0*dist0 + dist1*dist1;
          D->data[ii * n + jj] = sum;
        }
      }
    }
    // remaining columns
    for (; j < n; j++) {
      for (int ii = i; ii < i+block_size; ii++) {
        // Euclidean distance
        double dist0 = X->data[ii * m] - X->data[j * m];
        double dist1 = X->data[ii * m + 1] - X->data[j * m + 1];
        double sum = dist0*dist0 + dist1*dist1;
        D->data[ii * n + j] = sum;
      }
    }
  }
  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Calculate distance in blocks of 128 points.
*/
void euclidean_dist_low_block128(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  int block_size = 128;

  // calculate non-diagonal entries
  int i = 0;
  for (; i < block_size*(n/block_size); i+=block_size) {
    int j = i;
    for (; j < block_size*(n/block_size); j+=block_size) {
      // full block
      for (int ii = i; ii < i+block_size; ii++) {
        int jj = (ii+1 > j) ? ii+1 : j;
        for (; jj < j+block_size; jj++) {
          // Euclidean distance
          double dist0 = X->data[ii * m] - X->data[jj * m];
          double dist1 = X->data[ii * m + 1] - X->data[jj * m + 1];
          double sum = dist0*dist0 + dist1*dist1;
          D->data[ii * n + jj] = sum;
        }
      }
    }
    // remaining columns
    for (; j < n; j++) {
      for (int ii = i; ii < i+block_size; ii++) {
        // Euclidean distance
        double dist0 = X->data[ii * m] - X->data[j * m];
        double dist1 = X->data[ii * m + 1] - X->data[j * m + 1];
        double sum = dist0*dist0 + dist1*dist1;
        D->data[ii * n + j] = sum;
      }
    }
  }
  // remaining elements
  for (; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      // Euclidean distance
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}