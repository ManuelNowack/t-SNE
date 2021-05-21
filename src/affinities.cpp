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

void affinities_unroll_fst(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
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

void affinities_vectorized(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {

  int n = Y->nrows;

  // calculate squared Euclidean distances
  MY_EUCLIDEAN_DIST(Y, D);

  // unnormalised perplexities
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      sum += value;
    }
  }
  // normalise
  double norm = 0.5 / sum;  // 1 / (2 * sum) because triangular matrix
  __m256d _norm = _mm256_set1_pd(norm);
  __m256d _min_prob = _mm256_set1_pd(kMinimumProbability);
  __m256d _a, _b, _c, _d;
  for (int i = 0; i < n; i++) {
    // begin and end of 32-byte aligned addresses after i
    int begin = (i + 16) / 16 * 16;
    int end = n / 16 * 16;
    for (int j = begin; j < end; j += 16) {
      _a = _mm256_load_pd(Q_numerators->data + i * n + j);
      _b = _mm256_load_pd(Q_numerators->data + i * n + j + 4);
      _c = _mm256_load_pd(Q_numerators->data + i * n + j + 8);
      _d = _mm256_load_pd(Q_numerators->data + i * n + j + 12);
      _a = _mm256_mul_pd(_a, _norm);
      _b = _mm256_mul_pd(_b, _norm);
      _c = _mm256_mul_pd(_c, _norm);
      _d = _mm256_mul_pd(_d, _norm);
      _a = _mm256_max_pd(_a, _min_prob);
      _b = _mm256_max_pd(_b, _min_prob);
      _c = _mm256_max_pd(_c, _min_prob);
      _d = _mm256_max_pd(_d, _min_prob);
      _mm256_store_pd(Q->data + i * n + j, _a);
      _mm256_store_pd(Q->data + i * n + j + 4, _b);
      _mm256_store_pd(Q->data + i * n + j + 8, _c);
      _mm256_store_pd(Q->data + i * n + j + 12, _d);
    }
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_accumulator(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {

  int n = Y->nrows;

  // calculate squared Euclidean distances
  MY_EUCLIDEAN_DIST(Y, D);

  // unnormalised perplexities
  double acc[4] = {};
  for (int i = 0; i < n; i++) {
    // begin and end of 32-byte aligned addresses after i
    int begin = (i + 4) / 4 * 4;
    int end = n / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      acc[0] += value;
    }
    for (int j = begin; j < end; j += 4) {
      double val[4];
      val[0] = 1 / (1 + D->data[i * n + j]);
      val[1] = 1 / (1 + D->data[i * n + j + 1]);
      val[2] = 1 / (1 + D->data[i * n + j + 2]);
      val[3] = 1 / (1 + D->data[i * n + j + 3]);
      Q_numerators->data[i * n + j] = val[0];
      Q_numerators->data[i * n + j + 1] = val[1];
      Q_numerators->data[i * n + j + 2] = val[2];
      Q_numerators->data[i * n + j + 3] = val[3];
      acc[0] += val[0];
      acc[1] += val[1];
      acc[2] += val[2];
      acc[3] += val[3];
    }
    for (int j = end; j < n; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      acc[0] += value;
    }
  }
  double sum = acc[0] + acc[1] + acc[2] + acc[3];
  // normalise
  double norm = 0.5 / sum;  // 1 / (2 * sum) because triangular matrix
  __m256d _norm = _mm256_set1_pd(norm);
  __m256d _min_prob = _mm256_set1_pd(kMinimumProbability);
  __m256d _a, _b, _c, _d;
  for (int i = 0; i < n; i++) {
    // begin and end of 32-byte aligned addresses after i
    int begin = (i + 16) / 16 * 16;
    int end = n / 16 * 16;
    for (int j = begin; j < end; j += 16) {
      _a = _mm256_load_pd(Q_numerators->data + i * n + j);
      _b = _mm256_load_pd(Q_numerators->data + i * n + j + 4);
      _c = _mm256_load_pd(Q_numerators->data + i * n + j + 8);
      _d = _mm256_load_pd(Q_numerators->data + i * n + j + 12);
      _a = _mm256_mul_pd(_a, _norm);
      _b = _mm256_mul_pd(_b, _norm);
      _c = _mm256_mul_pd(_c, _norm);
      _d = _mm256_mul_pd(_d, _norm);
      _a = _mm256_max_pd(_a, _min_prob);
      _b = _mm256_max_pd(_b, _min_prob);
      _c = _mm256_max_pd(_c, _min_prob);
      _d = _mm256_max_pd(_d, _min_prob);
      _mm256_store_pd(Q->data + i * n + j, _a);
      _mm256_store_pd(Q->data + i * n + j + 4, _b);
      _mm256_store_pd(Q->data + i * n + j + 8, _c);
      _mm256_store_pd(Q->data + i * n + j + 12, _d);
    }
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_accumulator_vectorized(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {

  int n = Y->nrows;

  // calculate squared Euclidean distances
  MY_EUCLIDEAN_DIST(Y, D);

  // unnormalised perplexities
  double acc[4] = {};
  for (int i = 0; i < n; i++) {
    // begin and end of 32-byte aligned addresses after i
    int begin = (i + 4) / 4 * 4;
    int end = n / 4 * 4;
    for (int j = i + 1; j < begin; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      acc[0] += value;
    }
    __m256d _acc = _mm256_load_pd(acc);
    __m256d _one = _mm256_set1_pd(1.0);
    __m256d _a;
    for (int j = begin; j < end; j += 4) {
      _a = _mm256_load_pd(D->data + i * n + j);
      _a = _mm256_add_pd(_a, _one);
      _a = _mm256_div_pd(_one, _a);
      _acc = _mm256_add_pd(_acc, _a);
      _mm256_store_pd(Q_numerators->data + i * n + j, _a);
    }
    _mm256_store_pd(acc, _acc);
    for (int j = end; j < n; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      acc[0] += value;
    }
  }
  double sum = acc[0] + acc[1] + acc[2] + acc[3];
  // normalise
  double norm = 0.5 / sum;  // 1 / (2 * sum) because triangular matrix
  __m256d _norm = _mm256_set1_pd(norm);
  __m256d _min_prob = _mm256_set1_pd(kMinimumProbability);
  __m256d _a, _b, _c, _d;
  for (int i = 0; i < n; i++) {
    // begin and end of 32-byte aligned addresses after i
    int begin = (i + 16) / 16 * 16;
    int end = n / 16 * 16;
    for (int j = begin; j < end; j += 16) {
      _a = _mm256_load_pd(Q_numerators->data + i * n + j);
      _b = _mm256_load_pd(Q_numerators->data + i * n + j + 4);
      _c = _mm256_load_pd(Q_numerators->data + i * n + j + 8);
      _d = _mm256_load_pd(Q_numerators->data + i * n + j + 12);
      _a = _mm256_mul_pd(_a, _norm);
      _b = _mm256_mul_pd(_b, _norm);
      _c = _mm256_mul_pd(_c, _norm);
      _d = _mm256_mul_pd(_d, _norm);
      _a = _mm256_max_pd(_a, _min_prob);
      _b = _mm256_max_pd(_b, _min_prob);
      _c = _mm256_max_pd(_c, _min_prob);
      _d = _mm256_max_pd(_d, _min_prob);
      _mm256_store_pd(Q->data + i * n + j, _a);
      _mm256_store_pd(Q->data + i * n + j + 4, _b);
      _mm256_store_pd(Q->data + i * n + j + 8, _c);
      _mm256_store_pd(Q->data + i * n + j + 12, _d);
    }
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}

void affinities_accumulator_fully_vectorized(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {

  int n = Y->nrows;

  // calculate squared Euclidean distances
  MY_EUCLIDEAN_DIST(Y, D);

  // unnormalised perplexities
  double acc[16] = {};
  for (int i = 0; i < n; i++) {
    // begin and end of 32-byte aligned addresses after i
    int begin = (i + 16) / 16 * 16;
    int end = n / 16 * 16;
    for (int j = i + 1; j < begin; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      acc[0] += value;
    }
    __m256d _acc_a = _mm256_load_pd(acc);
    __m256d _acc_b = _mm256_load_pd(acc + 4);
    __m256d _acc_c = _mm256_load_pd(acc + 8);
    __m256d _acc_d = _mm256_load_pd(acc + 12);
    __m256d _one = _mm256_set1_pd(1.0);
    __m256d _a, _b, _c, _d;
    for (int j = begin; j < end; j += 16) {
      _a = _mm256_load_pd(D->data + i * n + j);
      _b = _mm256_load_pd(D->data + i * n + j + 4);
      _c = _mm256_load_pd(D->data + i * n + j + 8);
      _d = _mm256_load_pd(D->data + i * n + j + 12);
      _a = _mm256_add_pd(_a, _one);
      _b = _mm256_add_pd(_b, _one);
      _c = _mm256_add_pd(_c, _one);
      _d = _mm256_add_pd(_d, _one);
      _a = _mm256_div_pd(_one, _a);
      _b = _mm256_div_pd(_one, _b);
      _c = _mm256_div_pd(_one, _c);
      _d = _mm256_div_pd(_one, _d);
      _acc_a = _mm256_add_pd(_acc_a, _a);
      _acc_b = _mm256_add_pd(_acc_b, _b);
      _acc_c = _mm256_add_pd(_acc_c, _c);
      _acc_d = _mm256_add_pd(_acc_d, _d);
      _mm256_store_pd(Q_numerators->data + i * n + j, _a);
      _mm256_store_pd(Q_numerators->data + i * n + j + 4, _b);
      _mm256_store_pd(Q_numerators->data + i * n + j + 8, _c);
      _mm256_store_pd(Q_numerators->data + i * n + j + 12, _d);
    }
    _mm256_store_pd(acc, _acc_a);
    _mm256_store_pd(acc + 4, _acc_b);
    _mm256_store_pd(acc + 8, _acc_c);
    _mm256_store_pd(acc + 12, _acc_d);
    for (int j = end; j < n; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      acc[0] += value;
    }
  }
  double sum = 0;
  for (int i = 0; i < 16; ++i) {
    sum += acc[i];
  }
  // normalise
  double norm = 0.5 / sum;  // 1 / (2 * sum) because triangular matrix
  __m256d _norm = _mm256_set1_pd(norm);
  __m256d _min_prob = _mm256_set1_pd(kMinimumProbability);
  __m256d _a, _b, _c, _d;
  for (int i = 0; i < n; i++) {
    // begin and end of 32-byte aligned addresses after i
    int begin = (i + 16) / 16 * 16;
    int end = n / 16 * 16;
    for (int j = begin; j < end; j += 16) {
      _a = _mm256_load_pd(Q_numerators->data + i * n + j);
      _b = _mm256_load_pd(Q_numerators->data + i * n + j + 4);
      _c = _mm256_load_pd(Q_numerators->data + i * n + j + 8);
      _d = _mm256_load_pd(Q_numerators->data + i * n + j + 12);
      _a = _mm256_mul_pd(_a, _norm);
      _b = _mm256_mul_pd(_b, _norm);
      _c = _mm256_mul_pd(_c, _norm);
      _d = _mm256_mul_pd(_d, _norm);
      _a = _mm256_max_pd(_a, _min_prob);
      _b = _mm256_max_pd(_b, _min_prob);
      _c = _mm256_max_pd(_c, _min_prob);
      _d = _mm256_max_pd(_d, _min_prob);
      _mm256_store_pd(Q->data + i * n + j, _a);
      _mm256_store_pd(Q->data + i * n + j + 4, _b);
      _mm256_store_pd(Q->data + i * n + j + 8, _c);
      _mm256_store_pd(Q->data + i * n + j + 12, _d);
    }
    for (int j = i + 1; j < begin; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
    for (int j = end; j < n; j++) {
      double value = Q_numerators->data[i * n + j] * norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      Q->data[i * n + j] = value;
    }
  }
}
