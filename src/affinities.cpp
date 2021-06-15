#include <immintrin.h>

#include "tsne/debug.h"
#include "tsne/hyperparams.h"
#include "tsne/matrix.h"
#include <tsne/func_registry.h>


#define TRANSPOSE_4X4(a, b, c, d, at, bt, ct, dt) \
__m256d ab_lo = _mm256_unpacklo_pd(a, b);\
__m256d ab_hi = _mm256_unpackhi_pd(a, b);\
__m256d cd_lo = _mm256_unpacklo_pd(c, d);\
__m256d cd_hi = _mm256_unpackhi_pd(c, d);\
__m256d ab_lo_swap = _mm256_permute4x64_pd(ab_lo, 0b01001110);\
__m256d ab_hi_swap = _mm256_permute4x64_pd(ab_hi, 0b01001110);\
__m256d cd_lo_swap = _mm256_permute4x64_pd(cd_lo, 0b01001110);\
__m256d cd_hi_swap = _mm256_permute4x64_pd(cd_hi, 0b01001110);\
at = _mm256_blend_pd(ab_lo, cd_lo_swap, 0b1100);\
bt = _mm256_blend_pd(ab_hi, cd_hi_swap, 0b1100);\
ct = _mm256_blend_pd(cd_lo, ab_lo_swap, 0b0011);\
dt = _mm256_blend_pd(cd_hi, ab_hi_swap, 0b0011);

void euclidean_dist_baseline(Matrix *X, Matrix *D);

#define MY_EUCLIDEAN_DIST(Y, D) euclidean_dist_baseline(Y, D)
// #define MY_EUCLIDEAN_DIST(Y, D)


void affinities_baseline(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  // unnormalised perplexities
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      Q_numerators->data[j * n + i] = value;
      sum += value;
    }
  }

  // set diagonal elements
  for (int i = 0; i < n; i++) {
    Q->data[i * n + i] = 0;
  }

  // because triangular matrix
  sum *= 2.0;

  double norm = 1.0 / sum;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
      Q->data[j * n + i] = value;
    }
  }
}

void affinities_no_triangle(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
  }

  double norm = 0.5 / upper_sum;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

// incorrect output but reduces transferred bytes by 25%
void affinities_no_Q_numerators(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q->data[i * n + j] = value;
      upper_sum += value;
    }
  }

  double norm = 0.5 / upper_sum;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = Q->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_unroll_fst_4(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum = 0.0;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;  // first 32-byte aligned address after i
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
    // the bottleneck is division with latency 14 and gap 4 -> unroll by 4
    for (int j = begin; j < end; j += 4) {
      double value_1 = 1.0 / (1 + D->data[i * n + j]);
      double value_2 = 1.0 / (1 + D->data[i * n + j + 1]);
      double value_3 = 1.0 / (1 + D->data[i * n + j + 2]);
      double value_4 = 1.0 / (1 + D->data[i * n + j + 3]);
      Q_numerators->data[i * n + j] = value_1;
      Q_numerators->data[i * n + j + 1] = value_2;
      Q_numerators->data[i * n + j + 2] = value_3;
      Q_numerators->data[i * n + j + 3] = value_4;
      upper_sum += value_1;
      upper_sum += value_2;
      upper_sum += value_3;
      upper_sum += value_4;
    }
    for (int j = end; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
  }

  double norm = 0.5 / upper_sum;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_unroll_snd_4(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
  }

  double norm = 0.5 / upper_sum;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;  // first 32-byte aligned address after i
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    // the bottleneck is multiplication with latency 4 and gap 0.5 -> unroll by 8?
    for (int j = begin; j < end; j += 4) {
      double value_1 = Q_numerators->data[i * n + j];
      double value_2 = Q_numerators->data[i * n + j + 1];
      double value_3 = Q_numerators->data[i * n + j + 2];
      double value_4 = Q_numerators->data[i * n + j + 3];
      value_1 *= norm;
      value_2 *= norm;
      value_3 *= norm;
      value_4 *= norm;
      if (value_1 < kMinimumProbability) {
        value_1 = kMinimumProbability;
      }
      if (value_2 < kMinimumProbability) {
        value_2 = kMinimumProbability;
      }
      if (value_3 < kMinimumProbability) {
        value_3 = kMinimumProbability;
      }
      if (value_4 < kMinimumProbability) {
        value_4 = kMinimumProbability;
      }
      Q->data[i * n + j] = value_1;
      Q->data[i * n + j + 1] = value_2;
      Q->data[i * n + j + 2] = value_3;
      Q->data[i * n + j + 3] = value_4;
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_unroll_snd_8(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
  }

  double norm = 0.5 / upper_sum;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;  // first 32-byte aligned address after i
    int end = begin + (n - begin) / 8 * 8;
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    // the bottleneck is multiplication with latency 4 and gap 0.5 -> unroll by 8
    for (int j = begin; j < end; j += 8) {
      double value_1 = Q_numerators->data[i * n + j];
      double value_2 = Q_numerators->data[i * n + j + 1];
      double value_3 = Q_numerators->data[i * n + j + 2];
      double value_4 = Q_numerators->data[i * n + j + 3];
      double value_5 = Q_numerators->data[i * n + j + 4];
      double value_6 = Q_numerators->data[i * n + j + 5];
      double value_7 = Q_numerators->data[i * n + j + 6];
      double value_8 = Q_numerators->data[i * n + j + 7];
      value_1 *= norm;
      value_2 *= norm;
      value_3 *= norm;
      value_4 *= norm;
      value_5 *= norm;
      value_6 *= norm;
      value_7 *= norm;
      value_8 *= norm;
      if (value_1 < kMinimumProbability) {
        value_1 = kMinimumProbability;
      }
      if (value_2 < kMinimumProbability) {
        value_2 = kMinimumProbability;
      }
      if (value_3 < kMinimumProbability) {
        value_3 = kMinimumProbability;
      }
      if (value_4 < kMinimumProbability) {
        value_4 = kMinimumProbability;
      }
      if (value_5 < kMinimumProbability) {
        value_5 = kMinimumProbability;
      }
      if (value_6 < kMinimumProbability) {
        value_6 = kMinimumProbability;
      }
      if (value_7 < kMinimumProbability) {
        value_7 = kMinimumProbability;
      }
      if (value_8 < kMinimumProbability) {
        value_8 = kMinimumProbability;
      }
      Q->data[i * n + j] = value_1;
      Q->data[i * n + j + 1] = value_2;
      Q->data[i * n + j + 2] = value_3;
      Q->data[i * n + j + 3] = value_4;
      Q->data[i * n + j + 4] = value_5;
      Q->data[i * n + j + 5] = value_6;
      Q->data[i * n + j + 6] = value_7;
      Q->data[i * n + j + 7] = value_8;
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_unroll_both(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum = 0.0;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
    for (int j = begin; j < end; j += 4) {
      double value_1 = 1.0 / (1 + D->data[i * n + j]);
      double value_2 = 1.0 / (1 + D->data[i * n + j + 1]);
      double value_3 = 1.0 / (1 + D->data[i * n + j + 2]);
      double value_4 = 1.0 / (1 + D->data[i * n + j + 3]);
      Q_numerators->data[i * n + j] = value_1;
      Q_numerators->data[i * n + j + 1] = value_2;
      Q_numerators->data[i * n + j + 2] = value_3;
      Q_numerators->data[i * n + j + 3] = value_4;
      upper_sum += value_1;
      upper_sum += value_2;
      upper_sum += value_3;
      upper_sum += value_4;
    }
    for (int j = end; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
  }

  double norm = 0.5 / upper_sum;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = begin; j < end; j += 4) {
      double value_1 = Q_numerators->data[i * n + j];
      double value_2 = Q_numerators->data[i * n + j + 1];
      double value_3 = Q_numerators->data[i * n + j + 2];
      double value_4 = Q_numerators->data[i * n + j + 3];
      value_1 *= norm;
      value_2 *= norm;
      value_3 *= norm;
      value_4 *= norm;
      if (value_1 < kMinimumProbability) {
        value_1 = kMinimumProbability;
      }
      if (value_2 < kMinimumProbability) {
        value_2 = kMinimumProbability;
      }
      if (value_3 < kMinimumProbability) {
        value_3 = kMinimumProbability;
      }
      if (value_4 < kMinimumProbability) {
        value_4 = kMinimumProbability;
      }
      Q->data[i * n + j] = value_1;
      Q->data[i * n + j + 1] = value_2;
      Q->data[i * n + j + 2] = value_3;
      Q->data[i * n + j + 3] = value_4;
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_vectorization(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum_scalar = 0.0;
  __m256d one = _mm256_set1_pd(1.0);
  __m256d upper_sum = _mm256_setzero_pd();
  __m256d a;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum_scalar += value;
    }
    for (int j = begin; j < end; j += 4) {
      a = _mm256_load_pd(D->data + i * n + j);
      a = _mm256_add_pd(a, one);
      a = _mm256_div_pd(one, a);
      upper_sum = _mm256_add_pd(upper_sum, a);
      _mm256_store_pd(Q_numerators->data + i * n + j, a);
    }
    for (int j = end; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum_scalar += value;
    }
  }
  double tmp[4];
  _mm256_store_pd(tmp, upper_sum);
  upper_sum_scalar += tmp[0] + tmp[1] + tmp[2] + tmp[3];

  double norm_scalar = 0.5 / upper_sum_scalar;
  __m256d norm = _mm256_set1_pd(norm_scalar);
  __m256d min_prob = _mm256_set1_pd(kMinimumProbability);
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = begin; j < end; j += 4) {
      a = _mm256_load_pd(Q_numerators->data + i * n + j);
      a = _mm256_mul_pd(a, norm);
      a = _mm256_max_pd(a, min_prob);
      _mm256_store_pd(Q->data + i * n + j, a);
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

// incorrect output but reduces transferred bytes by 25%
void affinities_vectorization_no_Q_numerators(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum_scalar = 0.0;
  __m256d one = _mm256_set1_pd(1.0);
  __m256d upper_sum = _mm256_setzero_pd();
  __m256d a;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q->data[i * n + j] = value;
      upper_sum_scalar += value;
    }
    for (int j = begin; j < end; j += 4) {
      a = _mm256_load_pd(D->data + i * n + j);
      a = _mm256_add_pd(a, one);
      a = _mm256_div_pd(one, a);
      upper_sum = _mm256_add_pd(upper_sum, a);
      _mm256_store_pd(Q->data + i * n + j, a);
    }
    for (int j = end; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q->data[i * n + j] = value;
      upper_sum_scalar += value;
    }
  }
  double tmp[4];
  _mm256_store_pd(tmp, upper_sum);
  upper_sum_scalar += tmp[0] + tmp[1] + tmp[2] + tmp[3];

  double norm_scalar = 0.5 / upper_sum_scalar;
  __m256d norm = _mm256_set1_pd(norm_scalar);
  __m256d min_prob = _mm256_set1_pd(kMinimumProbability);
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = Q->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = begin; j < end; j += 4) {
      a = _mm256_load_pd(Q->data + i * n + j);
      a = _mm256_mul_pd(a, norm);
      a = _mm256_max_pd(a, min_prob);
      _mm256_store_pd(Q->data + i * n + j, a);
    }
    for (int j = end; j < n; j++) {
      double value = Q->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_vectorization_4(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum_scalar = 0.0;
  __m256d one = _mm256_set1_pd(1.0);
  __m256d upper_sum = _mm256_setzero_pd();
  __m256d a, b, c, d;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 16 * 16;
    for (int j = i + 1; j < begin; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum_scalar += value;
    }
    for (int j = begin; j < end; j += 16) {
      a = _mm256_load_pd(D->data + i * n + j);
      b = _mm256_load_pd(D->data + i * n + j + 4);
      c = _mm256_load_pd(D->data + i * n + j + 8);
      d = _mm256_load_pd(D->data + i * n + j + 12);
      a = _mm256_add_pd(a, one);
      b = _mm256_add_pd(b, one);
      c = _mm256_add_pd(c, one);
      d = _mm256_add_pd(d, one);
      a = _mm256_div_pd(one, a);
      b = _mm256_div_pd(one, b);
      c = _mm256_div_pd(one, c);
      d = _mm256_div_pd(one, d);
      upper_sum = _mm256_add_pd(upper_sum, a);
      upper_sum = _mm256_add_pd(upper_sum, b);
      upper_sum = _mm256_add_pd(upper_sum, c);
      upper_sum = _mm256_add_pd(upper_sum, d);
      _mm256_store_pd(Q_numerators->data + i * n + j, a);
      _mm256_store_pd(Q_numerators->data + i * n + j + 4, b);
      _mm256_store_pd(Q_numerators->data + i * n + j + 8, c);
      _mm256_store_pd(Q_numerators->data + i * n + j + 12, d);
    }
    for (int j = end; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum_scalar += value;
    }
  }
  double tmp[4];
  _mm256_store_pd(tmp, upper_sum);
  upper_sum_scalar += tmp[0] + tmp[1] + tmp[2] + tmp[3];

  double norm_scalar = 0.5 / upper_sum_scalar;
  __m256d norm = _mm256_set1_pd(norm_scalar);
  __m256d min_prob = _mm256_set1_pd(kMinimumProbability);
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 16 * 16;
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = begin; j < end; j += 16) {
      a = _mm256_load_pd(Q_numerators->data + i * n + j);
      b = _mm256_load_pd(Q_numerators->data + i * n + j + 4);
      c = _mm256_load_pd(Q_numerators->data + i * n + j + 8);
      d = _mm256_load_pd(Q_numerators->data + i * n + j + 12);
      a = _mm256_mul_pd(a, norm);
      b = _mm256_mul_pd(b, norm);
      c = _mm256_mul_pd(c, norm);
      d = _mm256_mul_pd(d, norm);
      a = _mm256_max_pd(a, min_prob);
      b = _mm256_max_pd(b, min_prob);
      c = _mm256_max_pd(c, min_prob);
      d = _mm256_max_pd(d, min_prob);
      _mm256_store_pd(Q->data + i * n + j, a);
      _mm256_store_pd(Q->data + i * n + j + 4, b);
      _mm256_store_pd(Q->data + i * n + j + 8, c);
      _mm256_store_pd(Q->data + i * n + j + 12, d);
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_accumulator(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  MY_EUCLIDEAN_DIST(Y, D);

  double upper_sum = 0.0;
  __m256d one = _mm256_set1_pd(1.0);
  __m256d acc_a = _mm256_setzero_pd();
  __m256d acc_b = _mm256_setzero_pd();
  __m256d acc_c = _mm256_setzero_pd();
  __m256d acc_d = _mm256_setzero_pd();
  __m256d a, b, c, d;
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 16 * 16;
    for (int j = i + 1; j < begin; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
    for (int j = begin; j < end; j += 16) {
      a = _mm256_load_pd(D->data + i * n + j);
      b = _mm256_load_pd(D->data + i * n + j + 4);
      c = _mm256_load_pd(D->data + i * n + j + 8);
      d = _mm256_load_pd(D->data + i * n + j + 12);
      a = _mm256_add_pd(a, one);
      b = _mm256_add_pd(b, one);
      c = _mm256_add_pd(c, one);
      d = _mm256_add_pd(d, one);
      a = _mm256_div_pd(one, a);
      b = _mm256_div_pd(one, b);
      c = _mm256_div_pd(one, c);
      d = _mm256_div_pd(one, d);
      acc_a = _mm256_add_pd(acc_a, a);
      acc_b = _mm256_add_pd(acc_b, b);
      acc_c = _mm256_add_pd(acc_c, c);
      acc_d = _mm256_add_pd(acc_d, d);
      _mm256_store_pd(Q_numerators->data + i * n + j, a);
      _mm256_store_pd(Q_numerators->data + i * n + j + 4, b);
      _mm256_store_pd(Q_numerators->data + i * n + j + 8, c);
      _mm256_store_pd(Q_numerators->data + i * n + j + 12, d);
    }
    for (int j = end; j < n; j++) {
      double value = 1.0 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      upper_sum += value;
    }
  }
  double tmp[4];
  _mm256_store_pd(tmp, acc_a);
  upper_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
  _mm256_store_pd(tmp, acc_b);
  upper_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
  _mm256_store_pd(tmp, acc_c);
  upper_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
  _mm256_store_pd(tmp, acc_d);
  upper_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];

  double norm_scalar = 0.5 / upper_sum;
  __m256d norm = _mm256_set1_pd(norm_scalar);
  __m256d min_prob = _mm256_set1_pd(kMinimumProbability);
  for (int i = 0; i < n; i++) {
    int begin = (i + 4) / 4 * 4;
    int end = begin + (n - begin) / 16 * 16;
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = begin; j < end; j += 16) {
      a = _mm256_load_pd(Q_numerators->data + i * n + j);
      b = _mm256_load_pd(Q_numerators->data + i * n + j + 4);
      c = _mm256_load_pd(Q_numerators->data + i * n + j + 8);
      d = _mm256_load_pd(Q_numerators->data + i * n + j + 12);
      a = _mm256_mul_pd(a, norm);
      b = _mm256_mul_pd(b, norm);
      c = _mm256_mul_pd(c, norm);
      d = _mm256_mul_pd(d, norm);
      a = _mm256_max_pd(a, min_prob);
      b = _mm256_max_pd(b, min_prob);
      c = _mm256_max_pd(c, min_prob);
      d = _mm256_max_pd(d, min_prob);
      _mm256_store_pd(Q->data + i * n + j, a);
      _mm256_store_pd(Q->data + i * n + j + 4, b);
      _mm256_store_pd(Q->data + i * n + j + 8, c);
      _mm256_store_pd(Q->data + i * n + j + 12, d);
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value *= norm_scalar;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

/*
* Affinities combined with euclidean dist and 4x4 unrolling.
* 4x4 instead of 4x8 to simplify implementation.
*/
void affinities_vec_unroll4x4(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {

  int n = Y->nrows;
  int m = Y->ncols;

  double *Y_data = Y->data;
  double *Q_data = Q->data;
  double *Q_numerators_data = Q_numerators->data;

  const __m256i index = _mm256_set_epi64x(6, 4, 2, 0);
  const __m256d onehalf_vec = _mm256_set1_pd(0.5);
  const __m256d one_vec = _mm256_set1_pd(1);
  const __m256d two_vec = _mm256_set1_pd(2);
  const __m256d zero_vec = _mm256_setzero_pd();

  __m256d sum = _mm256_setzero_pd();

  for (int i = 0; i < 4*(n/4); i+=4) {

    __m256d x00 = _mm256_broadcast_sd(Y_data + m*i);
    __m256d x01 = _mm256_broadcast_sd(Y_data + m*i + 1);

    __m256d x10 = _mm256_broadcast_sd(Y_data + m*i + 2);
    __m256d x11 = _mm256_broadcast_sd(Y_data + m*i + 3);

    __m256d x20 = _mm256_broadcast_sd(Y_data + m*i + 4);
    __m256d x21 = _mm256_broadcast_sd(Y_data + m*i + 5);

    __m256d x30 = _mm256_broadcast_sd(Y_data + m*i + 6);
    __m256d x31 = _mm256_broadcast_sd(Y_data + m*i + 7);


    // Diagonal block
    int j = i;
    __m256d y00 = _mm256_i64gather_pd(Y_data + m*j, index, 8);
    __m256d y01 = _mm256_i64gather_pd(Y_data + m*j + 1, index, 8);


    __m256d diff000 = _mm256_sub_pd(x00, y00);
    __m256d diff001 = _mm256_sub_pd(x01, y01);

    __m256d prod00 = _mm256_mul_pd(diff000, diff000);
    __m256d dists00 = _mm256_fmadd_pd(diff001, diff001, prod00);
    __m256d qnum00 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists00));
    qnum00 = _mm256_blend_pd(qnum00, zero_vec, 0b0001);
    sum = _mm256_fmadd_pd(qnum00, onehalf_vec, sum);
    _mm256_storeu_pd(Q_numerators_data + n*i + j, qnum00);


    __m256d diff100 = _mm256_sub_pd(x10, y00);
    __m256d diff101 = _mm256_sub_pd(x11, y01);

    __m256d prod10 = _mm256_mul_pd(diff100, diff100);
    __m256d dists10 = _mm256_fmadd_pd(diff101, diff101, prod10);
    __m256d qnum10 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists10));
    qnum10 = _mm256_blend_pd(qnum10, zero_vec, 0b0010);
    sum = _mm256_fmadd_pd(qnum10, onehalf_vec, sum);
    _mm256_storeu_pd(Q_numerators_data + n*i + n + j, qnum10);


    __m256d diff200 = _mm256_sub_pd(x20, y00);
    __m256d diff201 = _mm256_sub_pd(x21, y01);

    __m256d prod20 = _mm256_mul_pd(diff200, diff200);
    __m256d dists20 = _mm256_fmadd_pd(diff201, diff201, prod20);
    __m256d qnum20 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists20));
    qnum20 = _mm256_blend_pd(qnum20, zero_vec, 0b0100);
    sum = _mm256_fmadd_pd(qnum20, onehalf_vec, sum);
    _mm256_storeu_pd(Q_numerators_data + n*i + 2*n + j, qnum20);


    __m256d diff300 = _mm256_sub_pd(x30, y00);
    __m256d diff301 = _mm256_sub_pd(x31, y01);

    __m256d prod30 = _mm256_mul_pd(diff300, diff300);
    __m256d dists30 = _mm256_fmadd_pd(diff301, diff301, prod30);
    __m256d qnum30 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists30));
    qnum30 = _mm256_blend_pd(qnum30, zero_vec, 0b1000);
    sum = _mm256_fmadd_pd(qnum30, onehalf_vec, sum);
    _mm256_storeu_pd(Q_numerators_data + n*i + 3*n + j, qnum30);

    // Non-diagonal blocks
    j = i + 4;
    for (; j < 4*(n/4); j+=4) {

      y00 = _mm256_i64gather_pd(Y_data + m*j, index, 8);
      y01 = _mm256_i64gather_pd(Y_data + m*j + 1, index, 8);


      diff000 = _mm256_sub_pd(x00, y00);
      diff001 = _mm256_sub_pd(x01, y01);

      prod00 = _mm256_mul_pd(diff000, diff000);
      dists00 = _mm256_fmadd_pd(diff001, diff001, prod00);
      qnum00 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists00));
      sum = _mm256_add_pd(qnum00, sum);
      _mm256_storeu_pd(Q_numerators_data + n*i + j, qnum00);


      diff100 = _mm256_sub_pd(x10, y00);
      diff101 = _mm256_sub_pd(x11, y01);

      prod10 = _mm256_mul_pd(diff100, diff100);
      dists10 = _mm256_fmadd_pd(diff101, diff101, prod10);
      qnum10 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists10));
      sum = _mm256_add_pd(qnum10, sum);
      _mm256_storeu_pd(Q_numerators_data + n*i + n + j, qnum10);


      diff200 = _mm256_sub_pd(x20, y00);
      diff201 = _mm256_sub_pd(x21, y01);

      prod20 = _mm256_mul_pd(diff200, diff200);
      dists20 = _mm256_fmadd_pd(diff201, diff201, prod20);
      qnum20 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists20));
      sum = _mm256_add_pd(qnum20, sum);
      _mm256_storeu_pd(Q_numerators_data + n*i + 2*n + j, qnum20);


      diff300 = _mm256_sub_pd(x30, y00);
      diff301 = _mm256_sub_pd(x31, y01);

      prod30 = _mm256_mul_pd(diff300, diff300);
      dists30 = _mm256_fmadd_pd(diff301, diff301, prod30);
      qnum30 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists30));
      sum = _mm256_add_pd(qnum30, sum);
      _mm256_storeu_pd(Q_numerators_data + n*i + 3*n + j, qnum30);


      // Fill lower triangular 4x4 block
      __m256d qnum00t, qnum10t, qnum20t, qnum30t;
      TRANSPOSE_4X4(qnum00, qnum10, qnum20, qnum30, qnum00t, qnum10t, qnum20t, qnum30t);
      _mm256_storeu_pd(Q_numerators_data + n*j + i, qnum00t);
      _mm256_storeu_pd(Q_numerators_data + n*j + n + i, qnum10t);
      _mm256_storeu_pd(Q_numerators_data + n*j + 2*n + i, qnum20t);
      _mm256_storeu_pd(Q_numerators_data + n*j + 3*n + i, qnum30t);
    }
  }

  sum = _mm256_mul_pd(sum, two_vec); // Only upper triangular elements were summed


  // Normalize
  
  __m256d norm = _mm256_hadd_pd(sum, sum);
  norm = _mm256_add_pd(norm, _mm256_permute4x64_pd(norm, 0b01001110));
  norm = _mm256_div_pd(one_vec, norm);

  __m256d vec_min_prob = _mm256_set1_pd(kMinimumProbability);
  for (int i = 0; i < 4*(n/4); i+=4) {
    for (int j = i; j < 4*(n/4); j+=4) {
      __m256d q0 = _mm256_load_pd(Q_numerators_data + n*i + j);
      __m256d q1 = _mm256_load_pd(Q_numerators_data + n*i + n + j);
      __m256d q2 = _mm256_load_pd(Q_numerators_data + n*i + 2*n + j);
      __m256d q3 = _mm256_load_pd(Q_numerators_data + n*i + 3*n + j);

      q0 = _mm256_mul_pd(q0, norm);
      q1 = _mm256_mul_pd(q1, norm);
      q2 = _mm256_mul_pd(q2, norm);
      q3 = _mm256_mul_pd(q3, norm);

      q0 = _mm256_max_pd(q0, vec_min_prob);
      q1 = _mm256_max_pd(q1, vec_min_prob);
      q2 = _mm256_max_pd(q2, vec_min_prob);
      q3 = _mm256_max_pd(q3, vec_min_prob);

      _mm256_store_pd(Q_data + n*i + j, q0);
      _mm256_store_pd(Q_data + n*i + n + j, q1);
      _mm256_store_pd(Q_data + n*i + 2*n + j, q2);
      _mm256_store_pd(Q_data + n*i + 3*n + j, q3);

      // Fill lower triangular 4x4 block
      __m256d q0t, q1t, q2t, q3t;
      TRANSPOSE_4X4(q0, q1, q2, q3, q0t, q1t, q2t, q3t);
      _mm256_store_pd(Q_data + n*j + i, q0t);
      _mm256_store_pd(Q_data + n*j + n + i, q1t);
      _mm256_store_pd(Q_data + n*j + 2*n + i, q2t);
      _mm256_store_pd(Q_data + n*j + 3*n + i, q3t);
    }
  }
}