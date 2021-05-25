#include <float.h>
#include <immintrin.h>
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

/*
* Vectorization 1.
*/
void euclidean_dist_low_vec1(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = 2;

  // calculate non-diagonal entries
  for (int i = 0; i < 2*(n/2); i+=2) {
    __m256d x0 = _mm256_broadcast_pd((__m128d*)(X->data + m*i));
    __m256d x1 = _mm256_broadcast_pd((__m128d*)(X->data + m*i + m));

    for (int j = i; j < 2*(n/2); j+=2) {
      __m256d y = _mm256_load_pd(X->data + m*j);

      __m256d diff0 = _mm256_sub_pd(x0, y);
      __m256d diff1 = _mm256_sub_pd(x1, y);

      __m256d prod0 = _mm256_mul_pd(diff0, diff0);
      __m256d prod1 = _mm256_mul_pd(diff1, diff1);
      
      __m256d dists = _mm256_hadd_pd(prod0, prod1);
      dists = _mm256_permute4x64_pd(dists, 0b11011000);
      _mm256_storeu2_m128d(D->data + n*i + n + j, D->data + n*i + j, dists);
    }
  }
}

/*
* Vectorization 2.
*/
void euclidean_dist_low_vec2(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = 2;

  // calculate non-diagonal entries
  for (int i = 0; i < 2*(n/2); i+=2) {
    __m256d x = _mm256_load_pd(X->data + m*i);

    for (int j = i; j < 2*(n/2); j+=2) {
      __m256d y0 = _mm256_broadcast_pd((__m128d*)(X->data + m*j));
      __m256d y1 = _mm256_broadcast_pd((__m128d*)(X->data + m*j + m));

      __m256d diff0 = _mm256_sub_pd(x, y0);
      __m256d diff1 = _mm256_sub_pd(x, y1);

      __m256d prod0 = _mm256_mul_pd(diff0, diff0);
      __m256d prod1 = _mm256_mul_pd(diff1, diff1);
      
      __m256d dists = _mm256_hadd_pd(prod0, prod1);
      _mm256_storeu2_m128d(D->data + n*i + n + j, D->data + n*i + j, dists);
    }
  }
}

void _mm_print_pd(__m256d x) {
  __m256d y = x;
  for (int k = 0; k < 4; k++) {
    double val = _mm256_cvtsd_f64(y);
    printf("%.3e ", val);
    y = _mm256_permute4x64_pd(y, 0b00111001);
  }
  printf("\n");
}

void _mm_print_epi64(__m256i x) {
  __m256d y = _mm256_castsi256_pd(x);
  for (int k = 0; k < 4; k++) {
    double val = _mm256_cvtsd_f64(y);
    int *integer_val = (int *)&val;
    printf("%i ", *integer_val);
    y = _mm256_permute4x64_pd(y, 0b00111001);
  }
  printf("\n");
}

/*
* Vectorization 3.
*/
void euclidean_dist_low_vec3(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = 2;

  for (int j = 0; j < 4*(n/4); j+=4) {

    __m256i index = _mm256_set_epi64x(6, 4, 2, 0);
    __m256d y0 = _mm256_i64gather_pd(X->data + m*j, index, 8);
    __m256d y1 = _mm256_i64gather_pd(X->data + m*j + 1, index, 8);

    for (int i = 0; i < j+3; i++) {

      __m256d x0 = _mm256_broadcast_sd(X->data + m*i);
      __m256d x1 = _mm256_broadcast_sd(X->data + m*i + 1);

      __m256d diff0 = _mm256_sub_pd(x0, y0);
      __m256d diff1 = _mm256_sub_pd(x1, y1);

      __m256d prod = _mm256_mul_pd(diff0, diff0);
      __m256d dists = _mm256_fmadd_pd(diff1, diff1, prod);
      _mm256_store_pd(D->data + n*i + j, dists);
    }
  }
}

/*
* Vectorization 4.
*/
void euclidean_dist_low_vec4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = 2;

  for (int i = 0; i < n; i++) {
    __m256d x0 = _mm256_broadcast_sd(X->data + m*i);
    __m256d x1 = _mm256_broadcast_sd(X->data + m*i + 1);

    for (int j = 4*(i/4); j < 4*(n/4); j+=4) {
      __m256i index = _mm256_set_epi64x(6, 4, 2, 0);
      __m256d y0 = _mm256_i64gather_pd(X->data + m*j, index, 8);
      __m256d y1 = _mm256_i64gather_pd(X->data + m*j + 1, index, 8);

      __m256d diff0 = _mm256_sub_pd(x0, y0);
      __m256d diff1 = _mm256_sub_pd(x1, y1);

      __m256d prod = _mm256_mul_pd(diff0, diff0);
      __m256d dists = _mm256_fmadd_pd(diff1, diff1, prod);
      _mm256_store_pd(D->data + n*i + j, dists);
    }
  }
}

/*
* Vector unrolling by 2.
*/
void euclidean_dist_low_vec3_unroll2(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = 2;

  int j = 0;
  for (; j < 8*(n/8); j+=8) {

    __m256i index = _mm256_set_epi64x(6, 4, 2, 0);
    __m256d y00 = _mm256_i64gather_pd(X->data + m*j, index, 8);
    __m256d y01 = _mm256_i64gather_pd(X->data + m*j + 1, index, 8);
    __m256d y10 = _mm256_i64gather_pd(X->data + m*j + 8, index, 8);
    __m256d y11 = _mm256_i64gather_pd(X->data + m*j + 9, index, 8);

    for (int i = 0; i < j+7; i++) {

      __m256d x0 = _mm256_broadcast_sd(X->data + m*i);
      __m256d x1 = _mm256_broadcast_sd(X->data + m*i + 1);

      __m256d diff00 = _mm256_sub_pd(x0, y00);
      __m256d diff01 = _mm256_sub_pd(x1, y01);

      __m256d prod0 = _mm256_mul_pd(diff00, diff00);
      __m256d dists0 = _mm256_fmadd_pd(diff01, diff01, prod0);
      _mm256_store_pd(D->data + n*i + j, dists0);

      __m256d diff10 = _mm256_sub_pd(x0, y10);
      __m256d diff11 = _mm256_sub_pd(x1, y11);

      __m256d prod1 = _mm256_mul_pd(diff10, diff10);
      __m256d dists1 = _mm256_fmadd_pd(diff11, diff11, prod1);
      _mm256_store_pd(D->data + n*i + j + 4, dists1);
    }
  }
  for (; j < n; j++) {
    for (int i = 0; i < j; i++) {
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Vector unrolling by 4.
*/
void euclidean_dist_low_vec3_unroll4(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = 2;

  int j = 0;
  for (; j < 16*(n/16); j+=16) {

    __m256i index = _mm256_set_epi64x(6, 4, 2, 0);
    __m256d y00 = _mm256_i64gather_pd(X->data + m*j, index, 8);
    __m256d y01 = _mm256_i64gather_pd(X->data + m*j + 1, index, 8);
    __m256d y10 = _mm256_i64gather_pd(X->data + m*j + 8, index, 8);
    __m256d y11 = _mm256_i64gather_pd(X->data + m*j + 9, index, 8);
    __m256d y20 = _mm256_i64gather_pd(X->data + m*j + 16, index, 8);
    __m256d y21 = _mm256_i64gather_pd(X->data + m*j + 17, index, 8);
    __m256d y30 = _mm256_i64gather_pd(X->data + m*j + 24, index, 8);
    __m256d y31 = _mm256_i64gather_pd(X->data + m*j + 25, index, 8);

    for (int i = 0; i < j+15; i++) {

      __m256d x0 = _mm256_broadcast_sd(X->data + m*i);
      __m256d x1 = _mm256_broadcast_sd(X->data + m*i + 1);

      __m256d diff00 = _mm256_sub_pd(x0, y00);
      __m256d diff01 = _mm256_sub_pd(x1, y01);

      __m256d prod0 = _mm256_mul_pd(diff00, diff00);
      __m256d dists0 = _mm256_fmadd_pd(diff01, diff01, prod0);
      _mm256_store_pd(D->data + n*i + j, dists0);

      __m256d diff10 = _mm256_sub_pd(x0, y10);
      __m256d diff11 = _mm256_sub_pd(x1, y11);

      __m256d prod1 = _mm256_mul_pd(diff10, diff10);
      __m256d dists1 = _mm256_fmadd_pd(diff11, diff11, prod1);
      _mm256_store_pd(D->data + n*i + j + 4, dists1);

      __m256d diff20 = _mm256_sub_pd(x0, y20);
      __m256d diff21 = _mm256_sub_pd(x1, y21);

      __m256d prod2 = _mm256_mul_pd(diff20, diff20);
      __m256d dists2 = _mm256_fmadd_pd(diff21, diff21, prod2);
      _mm256_store_pd(D->data + n*i + j + 8, dists2);

      __m256d diff30 = _mm256_sub_pd(x0, y30);
      __m256d diff31 = _mm256_sub_pd(x1, y31);

      __m256d prod3 = _mm256_mul_pd(diff30, diff30);
      __m256d dists3 = _mm256_fmadd_pd(diff31, diff31, prod3);
      _mm256_store_pd(D->data + n*i + j + 12, dists3);
    }
  }
  for (; j < n; j++) {
    for (int i = 0; i < j; i++) {
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Vector unrolling by 8.
*/
void euclidean_dist_low_vec3_unroll8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = 2;

  int j = 0;
  for (; j < 32*(n/32); j+=32) {

    __m256i index = _mm256_set_epi64x(6, 4, 2, 0);
    __m256d y00 = _mm256_i64gather_pd(X->data + m*j, index, 8);
    __m256d y01 = _mm256_i64gather_pd(X->data + m*j + 1, index, 8);
    __m256d y10 = _mm256_i64gather_pd(X->data + m*j + 8, index, 8);
    __m256d y11 = _mm256_i64gather_pd(X->data + m*j + 9, index, 8);
    __m256d y20 = _mm256_i64gather_pd(X->data + m*j + 16, index, 8);
    __m256d y21 = _mm256_i64gather_pd(X->data + m*j + 17, index, 8);
    __m256d y30 = _mm256_i64gather_pd(X->data + m*j + 24, index, 8);
    __m256d y31 = _mm256_i64gather_pd(X->data + m*j + 25, index, 8);

    __m256d y40 = _mm256_i64gather_pd(X->data + m*j + 32, index, 8);
    __m256d y41 = _mm256_i64gather_pd(X->data + m*j + 33, index, 8);
    __m256d y50 = _mm256_i64gather_pd(X->data + m*j + 40, index, 8);
    __m256d y51 = _mm256_i64gather_pd(X->data + m*j + 41, index, 8);
    __m256d y60 = _mm256_i64gather_pd(X->data + m*j + 48, index, 8);
    __m256d y61 = _mm256_i64gather_pd(X->data + m*j + 49, index, 8);
    __m256d y70 = _mm256_i64gather_pd(X->data + m*j + 56, index, 8);
    __m256d y71 = _mm256_i64gather_pd(X->data + m*j + 57, index, 8);

    for (int i = 0; i < j+31; i++) {

      __m256d x0 = _mm256_broadcast_sd(X->data + m*i);
      __m256d x1 = _mm256_broadcast_sd(X->data + m*i + 1);

      __m256d diff00 = _mm256_sub_pd(x0, y00);
      __m256d diff01 = _mm256_sub_pd(x1, y01);

      __m256d prod0 = _mm256_mul_pd(diff00, diff00);
      __m256d dists0 = _mm256_fmadd_pd(diff01, diff01, prod0);
      _mm256_store_pd(D->data + n*i + j, dists0);

      __m256d diff10 = _mm256_sub_pd(x0, y10);
      __m256d diff11 = _mm256_sub_pd(x1, y11);

      __m256d prod1 = _mm256_mul_pd(diff10, diff10);
      __m256d dists1 = _mm256_fmadd_pd(diff11, diff11, prod1);
      _mm256_store_pd(D->data + n*i + j + 4, dists1);

      __m256d diff20 = _mm256_sub_pd(x0, y20);
      __m256d diff21 = _mm256_sub_pd(x1, y21);

      __m256d prod2 = _mm256_mul_pd(diff20, diff20);
      __m256d dists2 = _mm256_fmadd_pd(diff21, diff21, prod2);
      _mm256_store_pd(D->data + n*i + j + 8, dists2);

      __m256d diff30 = _mm256_sub_pd(x0, y30);
      __m256d diff31 = _mm256_sub_pd(x1, y31);

      __m256d prod3 = _mm256_mul_pd(diff30, diff30);
      __m256d dists3 = _mm256_fmadd_pd(diff31, diff31, prod3);
      _mm256_store_pd(D->data + n*i + j + 12, dists3);

      __m256d diff40 = _mm256_sub_pd(x0, y40);
      __m256d diff41 = _mm256_sub_pd(x1, y41);

      __m256d prod4 = _mm256_mul_pd(diff40, diff40);
      __m256d dists4 = _mm256_fmadd_pd(diff41, diff41, prod4);
      _mm256_store_pd(D->data + n*i + j + 16, dists4);

      __m256d diff50 = _mm256_sub_pd(x0, y50);
      __m256d diff51 = _mm256_sub_pd(x1, y51);

      __m256d prod5 = _mm256_mul_pd(diff50, diff50);
      __m256d dists5 = _mm256_fmadd_pd(diff51, diff51, prod5);
      _mm256_store_pd(D->data + n*i + j + 20, dists5);

      __m256d diff60 = _mm256_sub_pd(x0, y60);
      __m256d diff61 = _mm256_sub_pd(x1, y61);

      __m256d prod6 = _mm256_mul_pd(diff60, diff60);
      __m256d dists6 = _mm256_fmadd_pd(diff61, diff61, prod6);
      _mm256_store_pd(D->data + n*i + j + 24, dists6);

      __m256d diff70 = _mm256_sub_pd(x0, y70);
      __m256d diff71 = _mm256_sub_pd(x1, y71);

      __m256d prod7 = _mm256_mul_pd(diff70, diff70);
      __m256d dists7 = _mm256_fmadd_pd(diff71, diff71, prod7);
      _mm256_store_pd(D->data + n*i + j + 28, dists7);
    }
  }
  for (; j < n; j++) {
    for (int i = 0; i < j; i++) {
      double dist0 = X->data[i * m] - X->data[j * m];
      double dist1 = X->data[i * m + 1] - X->data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D->data[i * n + j] = sum;
    }
  }
}

/*
* Vector unrolling 4 by 8.
*/
void euclidean_dist_low_vec3_unroll4x8(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  double *X_data = X->data;
  double *D_data = D->data;

  __m256i index = _mm256_set_epi64x(6, 4, 2, 0);

  for (int i = 0; i < 4*(n/4); i+=4) {

    __m256d x00 = _mm256_broadcast_sd(X_data + m*i);
    __m256d x01 = _mm256_broadcast_sd(X_data + m*i + 1);

    __m256d x10 = _mm256_broadcast_sd(X_data + m*i + 2);
    __m256d x11 = _mm256_broadcast_sd(X_data + m*i + 3);

    __m256d x20 = _mm256_broadcast_sd(X_data + m*i + 4);
    __m256d x21 = _mm256_broadcast_sd(X_data + m*i + 5);

    __m256d x30 = _mm256_broadcast_sd(X_data + m*i + 6);
    __m256d x31 = _mm256_broadcast_sd(X_data + m*i + 7);

    int j = 8*(i/8);
    for (; j < 8*(n/8); j+=8) {

      __m256d y00 = _mm256_i64gather_pd(X_data + m*j, index, 8);
      __m256d y01 = _mm256_i64gather_pd(X_data + m*j + 1, index, 8);
      __m256d y10 = _mm256_i64gather_pd(X_data + m*j + 8, index, 8);
      __m256d y11 = _mm256_i64gather_pd(X_data + m*j + 9, index, 8);


      __m256d diff000 = _mm256_sub_pd(x00, y00);
      __m256d diff001 = _mm256_sub_pd(x01, y01);

      __m256d prod00 = _mm256_mul_pd(diff000, diff000);
      __m256d dists00 = _mm256_fmadd_pd(diff001, diff001, prod00);
      _mm256_storeu_pd(D_data + n*i + j, dists00);

      __m256d diff010 = _mm256_sub_pd(x00, y10);
      __m256d diff011 = _mm256_sub_pd(x01, y11);

      __m256d prod01 = _mm256_mul_pd(diff010, diff010);
      __m256d dists01 = _mm256_fmadd_pd(diff011, diff011, prod01);
      _mm256_storeu_pd(D_data + n*i + j + 4, dists01);


      __m256d diff100 = _mm256_sub_pd(x10, y00);
      __m256d diff101 = _mm256_sub_pd(x11, y01);

      __m256d prod10 = _mm256_mul_pd(diff100, diff100);
      __m256d dists10 = _mm256_fmadd_pd(diff101, diff101, prod10);
      _mm256_storeu_pd(D_data + n*i + n + j, dists10);

      __m256d diff110 = _mm256_sub_pd(x10, y10);
      __m256d diff111 = _mm256_sub_pd(x11, y11);

      __m256d prod11 = _mm256_mul_pd(diff110, diff110);
      __m256d dists11 = _mm256_fmadd_pd(diff111, diff111, prod11);
      _mm256_storeu_pd(D_data + n*i + n + j + 4, dists11);


      __m256d diff200 = _mm256_sub_pd(x20, y00);
      __m256d diff201 = _mm256_sub_pd(x21, y01);

      __m256d prod20 = _mm256_mul_pd(diff200, diff200);
      __m256d dists20 = _mm256_fmadd_pd(diff201, diff201, prod20);
      _mm256_storeu_pd(D_data + n*i + 2*n + j, dists20);

      __m256d diff210 = _mm256_sub_pd(x20, y10);
      __m256d diff211 = _mm256_sub_pd(x21, y11);

      __m256d prod21 = _mm256_mul_pd(diff210, diff210);
      __m256d dists21 = _mm256_fmadd_pd(diff211, diff211, prod21);
      _mm256_storeu_pd(D_data + n*i + 2*n + j + 4, dists21);


      __m256d diff300 = _mm256_sub_pd(x30, y00);
      __m256d diff301 = _mm256_sub_pd(x31, y01);

      __m256d prod30 = _mm256_mul_pd(diff300, diff300);
      __m256d dists30 = _mm256_fmadd_pd(diff301, diff301, prod30);
      _mm256_storeu_pd(D_data + n*i + 3*n + j, dists30);

      __m256d diff310 = _mm256_sub_pd(x30, y10);
      __m256d diff311 = _mm256_sub_pd(x31, y11);

      __m256d prod31 = _mm256_mul_pd(diff310, diff310);
      __m256d dists31 = _mm256_fmadd_pd(diff311, diff311, prod31);
      _mm256_storeu_pd(D_data + n*i + 3*n + j + 4, dists31);
    }

    for (; j < n; j++) {
      double dist0 = X_data[i * m] - X_data[j * m];
      double dist1 = X_data[i * m + 1] - X_data[j * m + 1];
      double sum = dist0*dist0 + dist1*dist1;
      D_data[i * n + j] = sum;

      dist0 = X_data[i * m + m] - X_data[j * m];
      dist1 = X_data[i * m + m + 1] - X_data[j * m + 1];
      sum = dist0*dist0 + dist1*dist1;
      D_data[i * n + n + j] = sum;

      dist0 = X_data[i * m + 2*m] - X_data[j * m];
      dist1 = X_data[i * m + 2*m + 1] - X_data[j * m + 1];
      sum = dist0*dist0 + dist1*dist1;
      D_data[i * n + 2*n + j] = sum;

      dist0 = X_data[i * m + 3*m] - X_data[j * m];
      dist1 = X_data[i * m + 3*m + 1] - X_data[j * m + 1];
      sum = dist0*dist0 + dist1*dist1;
      D_data[i * n + 3*n + j] = sum;
    }
  }
}