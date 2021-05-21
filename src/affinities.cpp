#include <immintrin.h>

#include "tsne/debug.h"
#include "tsne/hyperparams.h"
#include "tsne/matrix.h"

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

// incorrect output but reduces transfered bytes by 25%
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
    // the bottleneck is multiplication with latency 4 and gap 0.5 -> unroll by 8
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
