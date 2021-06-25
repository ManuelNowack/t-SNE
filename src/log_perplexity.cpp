#include <immintrin.h>
#include <math.h>
#include <tsne/debug.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#include <vectorclass/vectormath_exp.h>
#pragma GCC diagnostic pop

void log_perplexity_unroll2(double *distances, double *probabilities, int n,
                            int k, double precision, double *log_perplexity,
                            double *normlizer) {
  // calculate unnormalised conditional probabilities and normalization.
  double Z = 0, H = 0;
  int i = 0;
  for (; i < n - 1; i += 2) {
    double d0 = distances[i];
    double d1 = distances[i + 1];

    double p0 = (i == k) ? 0 : exp(-precision * d0);
    double p1 = (i + 1 == k) ? 0 : exp(-precision * d1);

    Z += p0 + p1;
    H += p0 * d0 + p1 * d1;

    probabilities[i] = p0;
    probabilities[i + 1] = p1;
  }

  for (; i < n; i++) {
    double di = distances[i];
    double pi = (i == k) ? 0 : exp(-precision * di);
    Z += pi;
    H += pi * di;
    probabilities[i] = pi;
  }

  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}

void log_perplexity_unroll4(double *distances, double *probabilities, int n,
                            int k, double precision, double *log_perplexity,
                            double *normlizer) {
  // calculate unnormalised conditional probabilities and normalization.
  double Z = 0, H = 0;
  int i = 0;
  for (; i < n - 3; i += 4) {
    double d0 = distances[i];
    double d1 = distances[i + 1];
    double d2 = distances[i + 2];
    double d3 = distances[i + 3];

    double p0 = (i == k) ? 0 : exp(-precision * d0);
    double p1 = (i + 1 == k) ? 0 : exp(-precision * d1);
    double p2 = (i + 2 == k) ? 0 : exp(-precision * d2);
    double p3 = (i + 3 == k) ? 0 : exp(-precision * d3);

    Z += p0 + p1 + p2 + p3;
    H += p0 * d0 + p1 * d1 + p2 * d2 + p3 * d3;

    probabilities[i] = p0;
    probabilities[i + 1] = p1;
    probabilities[i + 2] = p2;
    probabilities[i + 3] = p3;
  }

  for (; i < n; i++) {
    double di = distances[i];
    double pi = (i == k) ? 0 : exp(-precision * di);
    Z += pi;
    H += pi * di;
    probabilities[i] = pi;
  }

  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}

void log_perplexity_unroll8(double *distances, double *probabilities, int n,
                            int k, double precision, double *log_perplexity,
                            double *normlizer) {
  // calculate unnormalised conditional probabilities and normalization.
  double Z = 0, H = 0;
  int i = 0;
  for (; i < n - 7; i += 8) {
    double d0 = distances[i];
    double d1 = distances[i + 1];
    double d2 = distances[i + 2];
    double d3 = distances[i + 3];
    double d4 = distances[i + 4];
    double d5 = distances[i + 5];
    double d6 = distances[i + 6];
    double d7 = distances[i + 7];

    double p0 = (i == k) ? 0 : exp(-precision * d0);
    double p1 = (i + 1 == k) ? 0 : exp(-precision * d1);
    double p2 = (i + 2 == k) ? 0 : exp(-precision * d2);
    double p3 = (i + 3 == k) ? 0 : exp(-precision * d3);
    double p4 = (i + 4 == k) ? 0 : exp(-precision * d4);
    double p5 = (i + 5 == k) ? 0 : exp(-precision * d5);
    double p6 = (i + 6 == k) ? 0 : exp(-precision * d6);
    double p7 = (i + 7 == k) ? 0 : exp(-precision * d7);

    Z += p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
    H += p0 * d0 + p1 * d1 + p2 * d2 + p3 * d3 + p4 * d4 + p5 * d5 + p6 * d6 +
         p7 * d7;

    probabilities[i] = p0;
    probabilities[i + 1] = p1;
    probabilities[i + 2] = p2;
    probabilities[i + 3] = p3;
    probabilities[i + 4] = p4;
    probabilities[i + 5] = p5;
    probabilities[i + 6] = p6;
    probabilities[i + 7] = p7;
  }

  for (; i < n; i++) {
    double di = distances[i];
    double pi = (i == k) ? 0 : exp(-precision * di);
    Z += pi;
    H += pi * di;
    probabilities[i] = pi;
  }

  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}

// Add the elements of a packed double.
static inline double sum4d(__m256d acc) {
  __m128d acc_mm128d =
      _mm_add_pd(_mm256_extractf128_pd(acc, 1), _mm256_castpd256_pd128(acc));
  __m128d acc_sclr = _mm_add_sd(_mm_permute_pd(acc_mm128d, 1), acc_mm128d);
  return _mm_cvtsd_f64(acc_sclr);
}

void log_perplexity_avx(double *distances, double *probabilities, int n, int k,
                        double precision, double *log_perplexity,
                        double *normlizer) {
  __m256d neg_precision = _mm256_set1_pd(-precision);

  __m256i diag_index = _mm256_set1_epi64x(k);
  __m256i o0 = _mm256_set_epi64x(3, 2, 1, 0);  // offsets
  __m256i ii, m0;                              // masks

  __m256d z0 = _mm256_setzero_pd(), h0 = _mm256_setzero_pd();
  Vec4d e0;
  __m256d d0, p0;
  double Z, H;

  int i = 0;
  for (; i < n - 3; i += 4) {
    d0 = _mm256_load_pd(distances + i);
    e0 = _mm256_mul_pd(d0, neg_precision);

    ii = _mm256_set1_epi64x(i);
    m0 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o0), diag_index);
    p0 = _mm256_andnot_pd(_mm256_castsi256_pd(m0), exp(e0));

    d0 = _mm256_mul_pd(d0, p0);
    z0 = _mm256_add_pd(z0, p0);
    h0 = _mm256_add_pd(h0, d0);
    _mm256_store_pd(probabilities + i, p0);
  }

  Z = sum4d(z0);
  H = sum4d(h0);

  for (; i < n; i++) {
    double di = distances[i];
    double pi = (i == k) ? 0 : exp(-precision * di);
    Z += pi;
    H += pi * di;
    probabilities[i] = pi;
  }

  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}

void log_perplexity_avx_acc4(double *distances, double *probabilities, int n,
                             int k, double precision, double *log_perplexity,
                             double *normlizer) {
  __m256d neg_precision = _mm256_set1_pd(-precision);
  __m256d z0 = _mm256_setzero_pd();
  __m256d z1 = _mm256_setzero_pd();
  __m256d z2 = _mm256_setzero_pd();
  __m256d z3 = _mm256_setzero_pd();
  __m256d h0 = _mm256_setzero_pd();
  __m256d h1 = _mm256_setzero_pd();
  __m256d h2 = _mm256_setzero_pd();
  __m256d h3 = _mm256_setzero_pd();

  __m256i diag_index = _mm256_set1_epi64x(k);
  // offsets.
  __m256i o0 = _mm256_set_epi64x(3, 2, 1, 0);
  __m256i o1 = _mm256_set_epi64x(7, 6, 5, 4);
  __m256i o2 = _mm256_set_epi64x(11, 10, 9, 8);
  __m256i o3 = _mm256_set_epi64x(15, 14, 13, 12);
  __m256i ii, m0, m1, m2, m3;  // masks.

  Vec4d e0, e1, e2, e3;
  __m256d d0, d1, d2, d3, p0, p1, p2, p3;
  double Z, H;

  int i = 0;
  for (; i < n - 15; i += 16) {
    d0 = _mm256_load_pd(distances + i);
    d1 = _mm256_load_pd(distances + i + 4);
    d2 = _mm256_load_pd(distances + i + 8);
    d3 = _mm256_load_pd(distances + i + 12);

    e0 = _mm256_mul_pd(d0, neg_precision);
    e1 = _mm256_mul_pd(d1, neg_precision);
    e2 = _mm256_mul_pd(d2, neg_precision);
    e3 = _mm256_mul_pd(d3, neg_precision);

    ii = _mm256_set1_epi64x(i);
    m0 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o0), diag_index);
    m1 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o1), diag_index);
    m2 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o2), diag_index);
    m3 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o3), diag_index);

    p0 = _mm256_andnot_pd(_mm256_castsi256_pd(m0), exp(e0));
    p1 = _mm256_andnot_pd(_mm256_castsi256_pd(m1), exp(e1));
    p2 = _mm256_andnot_pd(_mm256_castsi256_pd(m2), exp(e2));
    p3 = _mm256_andnot_pd(_mm256_castsi256_pd(m3), exp(e3));

    d0 = _mm256_mul_pd(d0, p0);
    d1 = _mm256_mul_pd(d1, p1);
    d2 = _mm256_mul_pd(d2, p2);
    d3 = _mm256_mul_pd(d3, p3);

    z0 = _mm256_add_pd(z0, p0);
    z1 = _mm256_add_pd(z1, p1);
    z2 = _mm256_add_pd(z2, p2);
    z3 = _mm256_add_pd(z3, p3);

    h0 = _mm256_add_pd(h0, d0);
    h1 = _mm256_add_pd(h1, d1);
    h2 = _mm256_add_pd(h2, d2);
    h3 = _mm256_add_pd(h3, d3);

    _mm256_store_pd(probabilities + i, p0);
    _mm256_store_pd(probabilities + i + 4, p1);
    _mm256_store_pd(probabilities + i + 8, p2);
    _mm256_store_pd(probabilities + i + 12, p3);
  }

  Z = sum4d(z0) + sum4d(z1) + sum4d(z2) + sum4d(z3);
  H = sum4d(h0) + sum4d(h1) + sum4d(h2) + sum4d(h3);

  for (; i < n; i++) {
    double di = distances[i];
    double pi = exp(-precision * di);
    Z += pi;
    H += pi * di;
    probabilities[i] = pi;
  }

  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}

void log_perplexity_avx_fma_acc4(double *distances, double *probabilities,
                                 int n, int k, double precision,
                                 double *log_perplexity, double *normlizer) {
  __m256d neg_precision = _mm256_set1_pd(-precision);
  __m256d z0 = _mm256_setzero_pd();
  __m256d z1 = _mm256_setzero_pd();
  __m256d z2 = _mm256_setzero_pd();
  __m256d z3 = _mm256_setzero_pd();
  __m256d h0 = _mm256_setzero_pd();
  __m256d h1 = _mm256_setzero_pd();
  __m256d h2 = _mm256_setzero_pd();
  __m256d h3 = _mm256_setzero_pd();

  __m256i diag_index = _mm256_set1_epi64x(k);
  // offsets.
  __m256i o0 = _mm256_set_epi64x(3, 2, 1, 0);
  __m256i o1 = _mm256_set_epi64x(7, 6, 5, 4);
  __m256i o2 = _mm256_set_epi64x(11, 10, 9, 8);
  __m256i o3 = _mm256_set_epi64x(15, 14, 13, 12);
  __m256i ii, m0, m1, m2, m3;  // masks.

  Vec4d e0, e1, e2, e3;
  __m256d d0, d1, d2, d3, p0, p1, p2, p3;
  double Z, H;

  int i = 0;
  for (; i < n - 15; i += 16) {
    d0 = _mm256_load_pd(distances + i);
    d1 = _mm256_load_pd(distances + i + 4);
    d2 = _mm256_load_pd(distances + i + 8);
    d3 = _mm256_load_pd(distances + i + 12);

    e0 = _mm256_mul_pd(d0, neg_precision);
    e1 = _mm256_mul_pd(d1, neg_precision);
    e2 = _mm256_mul_pd(d2, neg_precision);
    e3 = _mm256_mul_pd(d3, neg_precision);

    ii = _mm256_set1_epi64x(i);
    m0 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o0), diag_index);
    m1 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o1), diag_index);
    m2 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o2), diag_index);
    m3 = _mm256_cmpeq_epi64(_mm256_add_epi64(ii, o3), diag_index);

    p0 = _mm256_andnot_pd(_mm256_castsi256_pd(m0), exp(e0));
    p1 = _mm256_andnot_pd(_mm256_castsi256_pd(m1), exp(e1));
    p2 = _mm256_andnot_pd(_mm256_castsi256_pd(m2), exp(e2));
    p3 = _mm256_andnot_pd(_mm256_castsi256_pd(m3), exp(e3));

    z0 = _mm256_add_pd(z0, p0);
    z1 = _mm256_add_pd(z1, p1);
    z2 = _mm256_add_pd(z2, p2);
    z3 = _mm256_add_pd(z3, p3);

    h0 = _mm256_fmadd_pd(d0, p0, h0);
    h1 = _mm256_fmadd_pd(d1, p1, h1);
    h2 = _mm256_fmadd_pd(d2, p2, h2);
    h3 = _mm256_fmadd_pd(d3, p3, h3);

    _mm256_store_pd(probabilities + i, p0);
    _mm256_store_pd(probabilities + i + 4, p1);
    _mm256_store_pd(probabilities + i + 8, p2);
    _mm256_store_pd(probabilities + i + 12, p3);
  }

  Z = sum4d(z0) + sum4d(z1) + sum4d(z2) + sum4d(z3);
  H = sum4d(h0) + sum4d(h1) + sum4d(h2) + sum4d(h3);

  for (; i < n; i++) {
    double di = distances[i];
    double pi = (i == k) ? 0 : exp(-precision * di);
    Z += pi;
    H += pi * di;
    probabilities[i] = pi;
  }

  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}
