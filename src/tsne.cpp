#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/debug.h>
#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

#define TRANSPOSE_4X4(a, b, c, d, at, bt, ct, dt)                \
  __m256d ab_lo = _mm256_unpacklo_pd(a, b);                      \
  __m256d ab_hi = _mm256_unpackhi_pd(a, b);                      \
  __m256d cd_lo = _mm256_unpacklo_pd(c, d);                      \
  __m256d cd_hi = _mm256_unpackhi_pd(c, d);                      \
  __m256d ab_lo_swap = _mm256_permute4x64_pd(ab_lo, 0b01001110); \
  __m256d ab_hi_swap = _mm256_permute4x64_pd(ab_hi, 0b01001110); \
  __m256d cd_lo_swap = _mm256_permute4x64_pd(cd_lo, 0b01001110); \
  __m256d cd_hi_swap = _mm256_permute4x64_pd(cd_hi, 0b01001110); \
  at = _mm256_blend_pd(ab_lo, cd_lo_swap, 0b1100);               \
  bt = _mm256_blend_pd(ab_hi, cd_hi_swap, 0b1100);               \
  ct = _mm256_blend_pd(cd_lo, ab_lo_swap, 0b0011);               \
  dt = _mm256_blend_pd(cd_hi, ab_hi_swap, 0b0011);

/*
 * Scalar combination
 */

/*
 * joint_probs_unroll8
 */
void _joint_probs(Matrix *X, Matrix *P, Matrix *D) {
  int n = X->nrows;

  euclidean_dist_alt_unroll4(X, D);

  double target_log_perplexity = log(kPerplexityTarget);

  // loop over all datapoints to determine precision and corresponding
  // probabilities
  for (int i = 0; i < n; i++) {
    double precision_min = 0.0;
    double precision_max = HUGE_VAL;
    double precision = 1;
    double *distances = &D->data[i * n];
    double *probabilities = &P->data[i * n];

    // bisection method for a fixed number of iterations
    double actual_log_perplexity, normalizer, diff;
    for (int iter = 0; iter < kJointProbsMaxIter; iter++) {
      log_perplexity_unroll8(distances, probabilities, n, i, precision,
                             &actual_log_perplexity, &normalizer);
      diff = actual_log_perplexity - target_log_perplexity;

      if (diff > 0) {
        // precision should be increased
        precision_min = precision;
        if (precision_max == HUGE_VAL) {
          precision *= 2;
        } else {
          precision = 0.5 * (precision + precision_max);
        }
      } else {
        // precision should be decreased
        precision_max = precision;
        if (precision_min == 0.0) {
          precision /= 2;
        } else {
          precision = 0.5 * (precision + precision_min);
        }
      }
    }

    // normalize probabilities
    for (int j = 0; j < n; j++) {
      probabilities[j] = probabilities[j] / normalizer;
    }
  }

  // convert conditional probabilties to joint probabilities
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double a = P->data[i * n + j];
      double b = P->data[j * n + i];
      double prob = (a + b) / (2 * n);

      // early exaggeration
      prob *= 4;

      // ensure minimal probability
      if (prob < 1e-12) prob = 1e-12;

      P->data[i * n + j] = prob;
      P->data[j * n + i] = prob;
    }
  }
}

/*
 * affinities_baseline
 */
void _affinities(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  euclidean_dist_low_unroll(Y, D);

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

/*
 * grad_desc_accumulators
 */
void _grad_desc(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum) {
  // calculate low-dimensional affinities
  _affinities(Y, &var->Q, &var->Q_numerators, &var->D);
  double *pdata = var->P.data;
  double *qdata = var->Q.data;
  double *q_numdata = var->Q_numerators.data;
  double *ydata = Y->data;
  double *gainsdata = var->gains.data;
  double *ydeltadata = var->Y_delta.data;

  // calculate gradient with respect to embeddings Y
  int twoi = 0;
  for (int i = 0; i < n; i++) {
    double value0 = 0;
    double value1 = 0;
    double value2 = 0;
    double value3 = 0;
    int twoj = 0;
    double ydatatwoi = ydata[twoi];
    double ydatatwoip1 = ydata[twoi + 1];
    double tmp1, tmp2;
    // try blocking approach?
    for (int j = 0; j < n; j += 2) {
      tmp1 = (pdata[i * n + j] - qdata[i * n + j]) * q_numdata[i * n + j];
      tmp2 = (pdata[i * n + j + 1] - qdata[i * n + j + 1]) *
             q_numdata[i * n + j + 1];
      value0 += tmp1 * (ydatatwoi - ydata[twoj]);
      value1 += tmp1 * (ydatatwoip1 - ydata[twoj + 1]);
      value2 += tmp2 * (ydatatwoi - ydata[twoj + 2]);
      value3 += tmp2 * (ydatatwoip1 - ydata[twoj + 3]);
      twoj += 4;
    }
    value0 += value2;
    value1 += value3;

    // calculate gains, according to adaptive heuristic of Python implementation
    double ydeltadata2i = ydeltadata[twoi];
    double ydeltadata2ip1 = ydeltadata[twoi + 1];
    bool positive_grad0 = ((value0) > 0);
    bool positive_delta0 = (ydeltadata2i > 0);
    bool positive_grad1 = ((value1) > 0);
    bool positive_delta1 = (ydeltadata2ip1 > 0);
    double val0 = gainsdata[twoi];
    double val1 = gainsdata[twoi + 1];

    val0 = (positive_grad0 == positive_delta0) ? val0 * 0.8 : val0 + 0.2;
    val1 = (positive_grad1 == positive_delta1) ? val1 * 0.8 : val1 + 0.2;
    if (val0 < kMinGain) {
      val0 = kMinGain;
    }
    if (val1 < kMinGain) {
      val1 = kMinGain;
    }

    gainsdata[twoi] = val0;
    gainsdata[twoi + 1] = val1;

    ydeltadata[twoi] = momentum * ydeltadata2i - fourkEta * val0 * value0;
    ydeltadata[twoi + 1] = momentum * ydeltadata2ip1 - fourkEta * val1 * value1;

    twoi += 2;
  }

  double mean0 = 0, mean1 = 0, mean2 = 0, mean3 = 0;
  int twon = 2 * n;
  for (int i = 0; i < twon; i += 4) {
    // update step
    ydata[i] += ydeltadata[i];
    ydata[i + 1] += ydeltadata[i + 1];
    ydata[i + 2] += ydeltadata[i + 2];
    ydata[i + 3] += ydeltadata[i + 3];
    mean0 += ydata[i];
    mean1 += ydata[i + 1];
    mean2 += ydata[i + 2];
    mean3 += ydata[i + 3];
  }
  // take mean
  mean0 += mean2;
  mean1 += mean3;
  mean0 /= n;
  mean1 /= n;
  // center
  for (int i = 0; i < twon; i += 4) {
    ydata[i] -= mean0;
    ydata[i + 1] -= mean1;
    ydata[i + 2] -= mean0;
    ydata[i + 3] -= mean1;
  }
}

void tsne_scalar(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim) {
  int n = X->nrows;

  // compute high level joint probabilities
  _joint_probs(X, &var->P, &var->D);

  // determine embeddings
  // initialisations
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      var->gains.data[i * n_dim + j] = 1;
    }
  }

  double momentum = kInitialMomentum;
  for (int iter = 0; iter < kGradDescMaxIter; iter++) {
    // early exaggeration only for first 100 iterations
    if (iter == 100) {
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double value = var->P.data[i * n + j] / 4;
          var->P.data[i * n + j] = value;
          var->P.data[j * n + i] = value;
        }
      }
    }

    // reduce momentum at iteration 20
    if (iter == 20) momentum = kFinalMomentum;

    _grad_desc(Y, var, n, n_dim, momentum);
  }
}

/*
 * Vectorized combination
 */

/*
 * joint_probs_avx_fma_acc4
 */
void _joint_probs_vec(Matrix *X, Matrix *P, Matrix *D) {
  int n = X->nrows;

  euclidean_dist_alt_vec_unroll4x4(X, D);

  double target_log_perplexity = log(kPerplexityTarget);

  // loop over all datapoints to determine precision and corresponding
  // probabilities
  for (int i = 0; i < n; i++) {
    double precision_min = 0.0;
    double precision_max = HUGE_VAL;
    double precision = 1;
    double *distances = &D->data[i * n];
    double *probabilities = &P->data[i * n];

    // bisection method for a fixed number of iterations
    double actual_log_perplexity, normalizer, diff;
    for (int iter = 0; iter < kJointProbsMaxIter; iter++) {
      log_perplexity_avx_fma_acc4(distances, probabilities, n, i, precision,
                                  &actual_log_perplexity, &normalizer);
      diff = actual_log_perplexity - target_log_perplexity;

      if (diff > 0) {
        // precision should be increased
        precision_min = precision;
        if (precision_max == HUGE_VAL) {
          precision *= 2;
        } else {
          precision = 0.5 * (precision + precision_max);
        }
      } else {
        // precision should be decreased
        precision_max = precision;
        if (precision_min == 0.0) {
          precision /= 2;
        } else {
          precision = 0.5 * (precision + precision_min);
        }
      }
    }

    // normalize probabilities
    for (int j = 0; j < n; j++) {
      probabilities[j] = probabilities[j] / normalizer;
    }
  }

  // convert conditional probabilties to joint probabilities
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double a = P->data[i * n + j];
      double b = P->data[j * n + i];
      double prob = (a + b) / (2 * n);

      // early exaggeration
      prob *= 4;

      // ensure minimal probability
      if (prob < 1e-12) prob = 1e-12;

      P->data[i * n + j] = prob;
      P->data[j * n + i] = prob;
    }
  }
}

/*
 * affinities_baseline
 */
void _affinities_vec(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {
  int n = Y->nrows;

  euclidean_dist_low_vec3_unroll4x8(Y, D);

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

/*
 * grad_desc_vectorized
 */
void _grad_desc_vec(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                    double momentum) {
  // calculate low-dimensional affinities
  _affinities_vec(Y, &var->Q, &var->Q_numerators, &var->D);

  const int ymask = 0b11011000;  // switch elements at pos 1 and 2
  double *pdata = var->P.data;
  double *qdata = var->Q.data;
  double *q_numdata = var->Q_numerators.data;
  double *ydata = Y->data;
  double *gainsdata = var->gains.data;
  double *ydeltadata = var->Y_delta.data;

  // calculate gradient with respect to embeddings Y
  int twoi = 0;
  __m256d zero = {0, 0, 0, 0};
  for (int i = 0; i < n; i += 4) {
    int twoj = 0;
    __m256d yleft, yright, y1, y2, yfixleft1, yfixleft2, yfixleft3, yfixleft4,
        yfixright1, yfixright2, yfixright3, yfixright4;
    __m256d valueleft1 = _mm256_setzero_pd();
    __m256d valueright1 = _mm256_setzero_pd();
    __m256d valueleft2 = _mm256_setzero_pd();
    __m256d valueright2 = _mm256_setzero_pd();
    __m256d valueleft3 = _mm256_setzero_pd();
    __m256d valueright3 = _mm256_setzero_pd();
    __m256d valueleft4 = _mm256_setzero_pd();
    __m256d valueright4 = _mm256_setzero_pd();
    yfixleft1 = _mm256_broadcast_sd(ydata + twoi);
    yfixleft2 = _mm256_broadcast_sd(ydata + twoi + 2);
    yfixleft3 = _mm256_broadcast_sd(ydata + twoi + 4);
    yfixleft4 = _mm256_broadcast_sd(ydata + twoi + 6);
    yfixright1 = _mm256_broadcast_sd(ydata + twoi + 1);
    yfixright2 = _mm256_broadcast_sd(ydata + twoi + 3);
    yfixright3 = _mm256_broadcast_sd(ydata + twoi + 5);
    yfixright4 = _mm256_broadcast_sd(ydata + twoi + 7);
    for (int j = 0; j < n; j += 4) {
      __m256d p1, p2, p3, p4, q1, q2, q3, q4, qnum1, qnum2, qnum3, qnum4, tmp1,
          tmp2, tmp3, tmp4;
      y1 = _mm256_load_pd(ydata + twoj);
      y2 = _mm256_load_pd(ydata + twoj + 4);
      // sort such that we have column wise 4 y elements
      yleft = _mm256_unpacklo_pd(y1, y2);
      yright = _mm256_unpackhi_pd(y1, y2);
      yleft = _mm256_permute4x64_pd(yleft, ymask);
      yright = _mm256_permute4x64_pd(yright, ymask);
      p1 = _mm256_load_pd(pdata + i * n + j);
      p2 = _mm256_load_pd(pdata + i * n + j + n);
      p3 = _mm256_load_pd(pdata + i * n + j + 2 * n);
      p4 = _mm256_load_pd(pdata + i * n + j + 3 * n);
      q1 = _mm256_load_pd(qdata + i * n + j);
      q2 = _mm256_load_pd(qdata + i * n + j + n);
      q3 = _mm256_load_pd(qdata + i * n + j + 2 * n);
      q4 = _mm256_load_pd(qdata + i * n + j + 3 * n);
      qnum1 = _mm256_load_pd(q_numdata + i * n + j);
      qnum2 = _mm256_load_pd(q_numdata + i * n + j + n);
      qnum3 = _mm256_load_pd(q_numdata + i * n + j + 2 * n);
      qnum4 = _mm256_load_pd(q_numdata + i * n + j + 3 * n);

      tmp1 = _mm256_mul_pd(_mm256_sub_pd(p1, q1), qnum1);
      tmp2 = _mm256_mul_pd(_mm256_sub_pd(p2, q2), qnum2);
      tmp3 = _mm256_mul_pd(_mm256_sub_pd(p3, q3), qnum3);
      tmp4 = _mm256_mul_pd(_mm256_sub_pd(p4, q4), qnum4);
      valueleft1 = _mm256_add_pd(
          _mm256_mul_pd(tmp1, _mm256_sub_pd(yfixleft1, yleft)), valueleft1);
      valueleft2 = _mm256_add_pd(
          _mm256_mul_pd(tmp2, _mm256_sub_pd(yfixleft2, yleft)), valueleft2);
      valueleft3 = _mm256_add_pd(
          _mm256_mul_pd(tmp3, _mm256_sub_pd(yfixleft3, yleft)), valueleft3);
      valueleft4 = _mm256_add_pd(
          _mm256_mul_pd(tmp4, _mm256_sub_pd(yfixleft4, yleft)), valueleft4);
      valueright1 = _mm256_add_pd(
          _mm256_mul_pd(tmp1, _mm256_sub_pd(yfixright1, yright)), valueright1);
      valueright2 = _mm256_add_pd(
          _mm256_mul_pd(tmp2, _mm256_sub_pd(yfixright2, yright)), valueright2);
      valueright3 = _mm256_add_pd(
          _mm256_mul_pd(tmp3, _mm256_sub_pd(yfixright3, yright)), valueright3);
      valueright4 = _mm256_add_pd(
          _mm256_mul_pd(tmp4, _mm256_sub_pd(yfixright4, yright)), valueright4);
      twoj += 8;
    }

    double *v = (double *)aligned_alloc(32, 16 * sizeof(double));
    _mm256_store_pd(v, _mm256_hadd_pd(valueleft1, valueright1));
    _mm256_store_pd(v + 4, _mm256_hadd_pd(valueleft2, valueright2));
    _mm256_store_pd(v + 8, _mm256_hadd_pd(valueleft3, valueright3));
    _mm256_store_pd(v + 12, _mm256_hadd_pd(valueleft4, valueright4));
    __m256d values_left, values_right;
    values_left = _mm256_set_pd(v[12] + v[14], v[8] + v[10], v[4] + v[6],
                                v[0] + v[2]);  // correct
    values_right = _mm256_set_pd(v[13] + v[15], v[9] + v[11], v[5] + v[7],
                                 v[1] + v[3]);  // correct

    __m256d ydeltaleft, ydeltaright, ydelta1, ydelta2, pos_grad_left,
        pos_grad_right, pos_delta_left, pos_delta_right, gainsleft, gainsright,
        gains0, gains1;
    ydelta1 = _mm256_load_pd(ydeltadata + twoi);
    ydelta2 = _mm256_load_pd(ydeltadata + twoi + 4);
    // sort such that we have column wise 4 ydelta elements
    ydeltaleft = _mm256_unpacklo_pd(ydelta1, ydelta2);
    ydeltaright = _mm256_unpackhi_pd(ydelta1, ydelta2);
    ydeltaleft = _mm256_permute4x64_pd(ydeltaleft, ymask);
    ydeltaright = _mm256_permute4x64_pd(ydeltaright, ymask);

    // compute boolean masks
    pos_grad_left = _mm256_cmp_pd(values_left, zero, _CMP_GT_OQ);
    pos_grad_right = _mm256_cmp_pd(values_right, zero, _CMP_GT_OQ);
    pos_delta_left = _mm256_cmp_pd(ydeltaleft, zero, _CMP_GT_OQ);
    pos_delta_right = _mm256_cmp_pd(ydeltaright, zero, _CMP_GT_OQ);

    // load gains
    gains0 = _mm256_load_pd(gainsdata + twoi);
    gains1 = _mm256_load_pd(gainsdata + twoi + 4);

    // sort gains into left and right
    gainsleft = _mm256_unpacklo_pd(gains0, gains1);
    gainsright = _mm256_unpackhi_pd(gains0, gains1);
    gainsleft = _mm256_permute4x64_pd(gainsleft, ymask);
    gainsright = _mm256_permute4x64_pd(gainsright, ymask);

    __m256d gainsmul_left, gainsmul_right, gainsplus_left, gainsplus_right,
        mask_left, mask_right;
    __m256d mulconst = {0.8, 0.8, 0.8, 0.8};
    __m256d addconst = {0.2, 0.2, 0.2, 0.2};
    gainsmul_left = _mm256_mul_pd(gainsleft, mulconst);
    gainsmul_right = _mm256_mul_pd(gainsright, mulconst);
    gainsplus_left = _mm256_add_pd(gainsleft, addconst);
    gainsplus_right = _mm256_add_pd(gainsright, addconst);
    mask_left = _mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_mm256_castpd_si256(pos_grad_left),
                           _mm256_castpd_si256(pos_delta_left)));
    mask_right = _mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_mm256_castpd_si256(pos_grad_right),
                           _mm256_castpd_si256(pos_delta_right)));

    gainsmul_left = _mm256_and_pd(mask_left, gainsmul_left);
    gainsmul_right = _mm256_and_pd(mask_right, gainsmul_right);
    gainsplus_left = _mm256_andnot_pd(mask_left, gainsplus_left);
    gainsplus_right = _mm256_andnot_pd(mask_right, gainsplus_right);

    gainsleft = _mm256_or_pd(gainsmul_left, gainsplus_left);
    gainsright = _mm256_or_pd(gainsmul_right, gainsplus_right);

    __m256d kmask_left, kmask_right;
    __m256d kmin = {kMinGain, kMinGain, kMinGain, kMinGain};
    kmask_left = _mm256_cmp_pd(gainsleft, kmin, _CMP_LT_OQ);
    kmask_right = _mm256_cmp_pd(gainsright, kmin, _CMP_LT_OQ);
    gainsleft = _mm256_blendv_pd(gainsleft, kmin, kmask_left);
    gainsright = _mm256_blendv_pd(gainsright, kmin, kmask_right);

    // unsort again
    gains0 = _mm256_permute4x64_pd(gainsleft, ymask);
    gains1 = _mm256_permute4x64_pd(gainsright, ymask);

    _mm256_store_pd(gainsdata + twoi, _mm256_unpacklo_pd(gains0, gains1));
    _mm256_store_pd(gainsdata + twoi + 4, _mm256_unpackhi_pd(gains0, gains1));

    __m256d momentum_v = {momentum, momentum, momentum, momentum};
    __m256d fourketa = {fourkEta, fourkEta, fourkEta, fourkEta};
    gainsleft = _mm256_mul_pd(fourketa, gainsleft);
    gainsright = _mm256_mul_pd(fourketa, gainsright);
    gainsleft = _mm256_mul_pd(gainsleft, values_left);
    gainsright = _mm256_mul_pd(gainsright, values_right);
    ydeltaleft = _mm256_fmsub_pd(momentum_v, ydeltaleft, gainsleft);
    ydeltaright = _mm256_fmsub_pd(momentum_v, ydeltaright, gainsright);

    ydelta1 = _mm256_permute4x64_pd(ydeltaleft, ymask);
    ydelta2 = _mm256_permute4x64_pd(ydeltaright, ymask);

    _mm256_store_pd(ydeltadata + twoi, _mm256_unpacklo_pd(ydelta1, ydelta2));
    _mm256_store_pd(ydeltadata + twoi + 4,
                    _mm256_unpackhi_pd(ydelta1, ydelta2));

    twoi += 8;
  }

  __m256d mean = {0, 0, 0, 0};
  __m256d n_vec = _mm256_set1_pd(n);
  __m256d y, ydelta, mean_shuffled, mean_shuffled2;
  const int mean_mask =
      0b10001101;  // setup such that we have [mean1 mean3 mean0 mean2]
  const int mean_mask2 =
      0b11011000;  // setup such that we have [mean0 mean2 mean1 mean3]
  const int mean_last_mask = 0b0110;
  int twon = 2 * n;
  for (int i = 0; i < twon; i += 4) {
    // load y, ydelta
    y = _mm256_load_pd(ydata + i);
    ydelta = _mm256_load_pd(ydeltadata + i);

    // update step
    y = _mm256_add_pd(y, ydelta);
    mean = _mm256_add_pd(mean, y);

    _mm256_store_pd(ydata + i, y);
  }
  mean_shuffled = _mm256_permute4x64_pd(mean, mean_mask);
  mean_shuffled2 = _mm256_permute4x64_pd(mean, mean_mask2);
  mean = _mm256_hadd_pd(
      mean_shuffled2,
      mean_shuffled);  // mean now holds [mean0 mean1 mean1 mean0]
  mean = _mm256_permute_pd(mean, mean_last_mask);
  // take mean
  mean = _mm256_div_pd(mean, n_vec);
  // center
  for (int i = 0; i < twon; i += 4) {
    y = _mm256_load_pd(ydata + i);
    y = _mm256_sub_pd(y, mean);
    _mm256_store_pd(ydata + i, y);
  }
}

void tsne_vec(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim) {
  int n = X->nrows;

  // compute high level joint probabilities
  // _joint_probs_vec(X, &var->P, &var->D);
  _joint_probs_vec(X, &var->P, &var->D);

  // determine embeddings
  // initialisations
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      var->gains.data[i * n_dim + j] = 1;
    }
  }

  double momentum = kInitialMomentum;
  for (int iter = 0; iter < kGradDescMaxIter; iter++) {
    // early exaggeration only for first 100 iterations
    if (iter == 100) {
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double value = var->P.data[i * n + j] / 4;
          var->P.data[i * n + j] = value;
          var->P.data[j * n + i] = value;
        }
      }
    }

    // reduce momentum at iteration 20
    if (iter == 20) momentum = kFinalMomentum;

    _grad_desc_vec(Y, var, n, n_dim, momentum);
  }
}

/*
 * Vectorized version 2
 * Using affinities_vec_unroll4x4
 */

void affinities_vec_unroll4x4(Matrix *Y, Matrix *Q, Matrix *Q_numerators,
                              Matrix *D);

/*
 * grad_desc_vectorized
 */
void _grad_desc_vec2(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                     double momentum) {
  // calculate low-dimensional affinities
  affinities_vec_unroll4x4(Y, &var->Q, &var->Q_numerators, &var->D);

  const int ymask = 0b11011000;  // switch elements at pos 1 and 2
  double *pdata = var->P.data;
  double *qdata = var->Q.data;
  double *q_numdata = var->Q_numerators.data;
  double *ydata = Y->data;
  double *gainsdata = var->gains.data;
  double *ydeltadata = var->Y_delta.data;

  // calculate gradient with respect to embeddings Y
  int twoi = 0;
  __m256d zero = {0, 0, 0, 0};
  for (int i = 0; i < n; i += 4) {
    int twoj = 0;
    __m256d yleft, yright, y1, y2, yfixleft1, yfixleft2, yfixleft3, yfixleft4,
        yfixright1, yfixright2, yfixright3, yfixright4;
    __m256d valueleft1 = _mm256_setzero_pd();
    __m256d valueright1 = _mm256_setzero_pd();
    __m256d valueleft2 = _mm256_setzero_pd();
    __m256d valueright2 = _mm256_setzero_pd();
    __m256d valueleft3 = _mm256_setzero_pd();
    __m256d valueright3 = _mm256_setzero_pd();
    __m256d valueleft4 = _mm256_setzero_pd();
    __m256d valueright4 = _mm256_setzero_pd();
    yfixleft1 = _mm256_broadcast_sd(ydata + twoi);
    yfixleft2 = _mm256_broadcast_sd(ydata + twoi + 2);
    yfixleft3 = _mm256_broadcast_sd(ydata + twoi + 4);
    yfixleft4 = _mm256_broadcast_sd(ydata + twoi + 6);
    yfixright1 = _mm256_broadcast_sd(ydata + twoi + 1);
    yfixright2 = _mm256_broadcast_sd(ydata + twoi + 3);
    yfixright3 = _mm256_broadcast_sd(ydata + twoi + 5);
    yfixright4 = _mm256_broadcast_sd(ydata + twoi + 7);
    for (int j = 0; j < n; j += 4) {
      __m256d p1, p2, p3, p4, q1, q2, q3, q4, qnum1, qnum2, qnum3, qnum4, tmp1,
          tmp2, tmp3, tmp4;
      y1 = _mm256_load_pd(ydata + twoj);
      y2 = _mm256_load_pd(ydata + twoj + 4);
      // sort such that we have column wise 4 y elements
      yleft = _mm256_unpacklo_pd(y1, y2);
      yright = _mm256_unpackhi_pd(y1, y2);
      yleft = _mm256_permute4x64_pd(yleft, ymask);
      yright = _mm256_permute4x64_pd(yright, ymask);
      p1 = _mm256_load_pd(pdata + i * n + j);
      p2 = _mm256_load_pd(pdata + i * n + j + n);
      p3 = _mm256_load_pd(pdata + i * n + j + 2 * n);
      p4 = _mm256_load_pd(pdata + i * n + j + 3 * n);
      q1 = _mm256_load_pd(qdata + i * n + j);
      q2 = _mm256_load_pd(qdata + i * n + j + n);
      q3 = _mm256_load_pd(qdata + i * n + j + 2 * n);
      q4 = _mm256_load_pd(qdata + i * n + j + 3 * n);
      qnum1 = _mm256_load_pd(q_numdata + i * n + j);
      qnum2 = _mm256_load_pd(q_numdata + i * n + j + n);
      qnum3 = _mm256_load_pd(q_numdata + i * n + j + 2 * n);
      qnum4 = _mm256_load_pd(q_numdata + i * n + j + 3 * n);

      tmp1 = _mm256_mul_pd(_mm256_sub_pd(p1, q1), qnum1);
      tmp2 = _mm256_mul_pd(_mm256_sub_pd(p2, q2), qnum2);
      tmp3 = _mm256_mul_pd(_mm256_sub_pd(p3, q3), qnum3);
      tmp4 = _mm256_mul_pd(_mm256_sub_pd(p4, q4), qnum4);
      valueleft1 = _mm256_add_pd(
          _mm256_mul_pd(tmp1, _mm256_sub_pd(yfixleft1, yleft)), valueleft1);
      valueleft2 = _mm256_add_pd(
          _mm256_mul_pd(tmp2, _mm256_sub_pd(yfixleft2, yleft)), valueleft2);
      valueleft3 = _mm256_add_pd(
          _mm256_mul_pd(tmp3, _mm256_sub_pd(yfixleft3, yleft)), valueleft3);
      valueleft4 = _mm256_add_pd(
          _mm256_mul_pd(tmp4, _mm256_sub_pd(yfixleft4, yleft)), valueleft4);
      valueright1 = _mm256_add_pd(
          _mm256_mul_pd(tmp1, _mm256_sub_pd(yfixright1, yright)), valueright1);
      valueright2 = _mm256_add_pd(
          _mm256_mul_pd(tmp2, _mm256_sub_pd(yfixright2, yright)), valueright2);
      valueright3 = _mm256_add_pd(
          _mm256_mul_pd(tmp3, _mm256_sub_pd(yfixright3, yright)), valueright3);
      valueright4 = _mm256_add_pd(
          _mm256_mul_pd(tmp4, _mm256_sub_pd(yfixright4, yright)), valueright4);
      twoj += 8;
    }

    double *v = (double *)aligned_alloc(32, 16 * sizeof(double));
    _mm256_store_pd(v, _mm256_hadd_pd(valueleft1, valueright1));
    _mm256_store_pd(v + 4, _mm256_hadd_pd(valueleft2, valueright2));
    _mm256_store_pd(v + 8, _mm256_hadd_pd(valueleft3, valueright3));
    _mm256_store_pd(v + 12, _mm256_hadd_pd(valueleft4, valueright4));
    __m256d values_left, values_right;
    values_left = _mm256_set_pd(v[12] + v[14], v[8] + v[10], v[4] + v[6],
                                v[0] + v[2]);  // correct
    values_right = _mm256_set_pd(v[13] + v[15], v[9] + v[11], v[5] + v[7],
                                 v[1] + v[3]);  // correct

    __m256d ydeltaleft, ydeltaright, ydelta1, ydelta2, pos_grad_left,
        pos_grad_right, pos_delta_left, pos_delta_right, gainsleft, gainsright,
        gains0, gains1;
    ydelta1 = _mm256_load_pd(ydeltadata + twoi);
    ydelta2 = _mm256_load_pd(ydeltadata + twoi + 4);
    // sort such that we have column wise 4 ydelta elements
    ydeltaleft = _mm256_unpacklo_pd(ydelta1, ydelta2);
    ydeltaright = _mm256_unpackhi_pd(ydelta1, ydelta2);
    ydeltaleft = _mm256_permute4x64_pd(ydeltaleft, ymask);
    ydeltaright = _mm256_permute4x64_pd(ydeltaright, ymask);

    // compute boolean masks
    pos_grad_left = _mm256_cmp_pd(values_left, zero, _CMP_GT_OQ);
    pos_grad_right = _mm256_cmp_pd(values_right, zero, _CMP_GT_OQ);
    pos_delta_left = _mm256_cmp_pd(ydeltaleft, zero, _CMP_GT_OQ);
    pos_delta_right = _mm256_cmp_pd(ydeltaright, zero, _CMP_GT_OQ);

    // load gains
    gains0 = _mm256_load_pd(gainsdata + twoi);
    gains1 = _mm256_load_pd(gainsdata + twoi + 4);

    // sort gains into left and right
    gainsleft = _mm256_unpacklo_pd(gains0, gains1);
    gainsright = _mm256_unpackhi_pd(gains0, gains1);
    gainsleft = _mm256_permute4x64_pd(gainsleft, ymask);
    gainsright = _mm256_permute4x64_pd(gainsright, ymask);

    __m256d gainsmul_left, gainsmul_right, gainsplus_left, gainsplus_right,
        mask_left, mask_right;
    __m256d mulconst = {0.8, 0.8, 0.8, 0.8};
    __m256d addconst = {0.2, 0.2, 0.2, 0.2};
    gainsmul_left = _mm256_mul_pd(gainsleft, mulconst);
    gainsmul_right = _mm256_mul_pd(gainsright, mulconst);
    gainsplus_left = _mm256_add_pd(gainsleft, addconst);
    gainsplus_right = _mm256_add_pd(gainsright, addconst);
    mask_left = _mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_mm256_castpd_si256(pos_grad_left),
                           _mm256_castpd_si256(pos_delta_left)));
    mask_right = _mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_mm256_castpd_si256(pos_grad_right),
                           _mm256_castpd_si256(pos_delta_right)));

    gainsmul_left = _mm256_and_pd(mask_left, gainsmul_left);
    gainsmul_right = _mm256_and_pd(mask_right, gainsmul_right);
    gainsplus_left = _mm256_andnot_pd(mask_left, gainsplus_left);
    gainsplus_right = _mm256_andnot_pd(mask_right, gainsplus_right);

    gainsleft = _mm256_or_pd(gainsmul_left, gainsplus_left);
    gainsright = _mm256_or_pd(gainsmul_right, gainsplus_right);

    __m256d kmask_left, kmask_right;
    __m256d kmin = {kMinGain, kMinGain, kMinGain, kMinGain};
    kmask_left = _mm256_cmp_pd(gainsleft, kmin, _CMP_LT_OQ);
    kmask_right = _mm256_cmp_pd(gainsright, kmin, _CMP_LT_OQ);
    gainsleft = _mm256_blendv_pd(gainsleft, kmin, kmask_left);
    gainsright = _mm256_blendv_pd(gainsright, kmin, kmask_right);

    // unsort again
    gains0 = _mm256_permute4x64_pd(gainsleft, ymask);
    gains1 = _mm256_permute4x64_pd(gainsright, ymask);

    _mm256_store_pd(gainsdata + twoi, _mm256_unpacklo_pd(gains0, gains1));
    _mm256_store_pd(gainsdata + twoi + 4, _mm256_unpackhi_pd(gains0, gains1));

    __m256d momentum_v = {momentum, momentum, momentum, momentum};
    __m256d fourketa = {fourkEta, fourkEta, fourkEta, fourkEta};
    gainsleft = _mm256_mul_pd(fourketa, gainsleft);
    gainsright = _mm256_mul_pd(fourketa, gainsright);
    gainsleft = _mm256_mul_pd(gainsleft, values_left);
    gainsright = _mm256_mul_pd(gainsright, values_right);
    ydeltaleft = _mm256_fmsub_pd(momentum_v, ydeltaleft, gainsleft);
    ydeltaright = _mm256_fmsub_pd(momentum_v, ydeltaright, gainsright);

    ydelta1 = _mm256_permute4x64_pd(ydeltaleft, ymask);
    ydelta2 = _mm256_permute4x64_pd(ydeltaright, ymask);

    _mm256_store_pd(ydeltadata + twoi, _mm256_unpacklo_pd(ydelta1, ydelta2));
    _mm256_store_pd(ydeltadata + twoi + 4,
                    _mm256_unpackhi_pd(ydelta1, ydelta2));

    twoi += 8;
  }

  __m256d mean = {0, 0, 0, 0};
  __m256d n_vec = _mm256_set1_pd(n);
  __m256d y, ydelta, mean_shuffled, mean_shuffled2;
  const int mean_mask =
      0b10001101;  // setup such that we have [mean1 mean3 mean0 mean2]
  const int mean_mask2 =
      0b11011000;  // setup such that we have [mean0 mean2 mean1 mean3]
  const int mean_last_mask = 0b0110;
  int twon = 2 * n;
  for (int i = 0; i < twon; i += 4) {
    // load y, ydelta
    y = _mm256_load_pd(ydata + i);
    ydelta = _mm256_load_pd(ydeltadata + i);

    // update step
    y = _mm256_add_pd(y, ydelta);
    mean = _mm256_add_pd(mean, y);

    _mm256_store_pd(ydata + i, y);
  }
  mean_shuffled = _mm256_permute4x64_pd(mean, mean_mask);
  mean_shuffled2 = _mm256_permute4x64_pd(mean, mean_mask2);
  mean = _mm256_hadd_pd(
      mean_shuffled2,
      mean_shuffled);  // mean now holds [mean0 mean1 mean1 mean0]
  mean = _mm256_permute_pd(mean, mean_last_mask);
  // take mean
  mean = _mm256_div_pd(mean, n_vec);
  // center
  for (int i = 0; i < twon; i += 4) {
    y = _mm256_load_pd(ydata + i);
    y = _mm256_sub_pd(y, mean);
    _mm256_store_pd(ydata + i, y);
  }
}

void tsne_vec2(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim) {
  int n = X->nrows;

  // compute high level joint probabilities
  // _joint_probs_vec(X, &var->P, &var->D);
  _joint_probs_vec(X, &var->P, &var->D);

  // determine embeddings
  // initialisations
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      var->gains.data[i * n_dim + j] = 1;
    }
  }

  double momentum = kInitialMomentum;
  for (int iter = 0; iter < kGradDescMaxIter; iter++) {
    // early exaggeration only for first 100 iterations
    if (iter == 100) {
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double value = var->P.data[i * n + j] / 4;
          var->P.data[i * n + j] = value;
          var->P.data[j * n + i] = value;
        }
      }
    }

    // reduce momentum at iteration 20
    if (iter == 20) momentum = kFinalMomentum;

    _grad_desc_vec2(Y, var, n, n_dim, momentum);
  }
}

/*
 * Vectorized version 3
 */

/*
 * Based on grad_desc_vectorized
 * Inline calculation of affinities
 * Postpone normalization of affinities to gradient update
 */
void _grad_desc_vec3(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                     double momentum) {
  const int ymask = 0b11011000;  // switch elements at pos 1 and 2
  double *pdata = var->P.data;
  double *q_numdata = var->Q_numerators.data;
  double *ydata = Y->data;
  double *gainsdata = var->gains.data;
  double *ydeltadata = var->Y_delta.data;

  // calculate unnormalized low-dimensional affinities
  __m256d norm;  // normalization factor required later
  {
    const __m256i index = _mm256_set_epi64x(6, 4, 2, 0);
    const __m256d onehalf_vec = _mm256_set1_pd(0.5);
    const __m256d one_vec = _mm256_set1_pd(1);
    const __m256d two_vec = _mm256_set1_pd(2);
    const __m256d zero_vec = _mm256_setzero_pd();

    __m256d sum = _mm256_setzero_pd();

    for (int i = 0; i < 4 * (n / 4); i += 4) {
      __m256d x00 = _mm256_broadcast_sd(ydata + n_dim * i);
      __m256d x01 = _mm256_broadcast_sd(ydata + n_dim * i + 1);

      __m256d x10 = _mm256_broadcast_sd(ydata + n_dim * i + 2);
      __m256d x11 = _mm256_broadcast_sd(ydata + n_dim * i + 3);

      __m256d x20 = _mm256_broadcast_sd(ydata + n_dim * i + 4);
      __m256d x21 = _mm256_broadcast_sd(ydata + n_dim * i + 5);

      __m256d x30 = _mm256_broadcast_sd(ydata + n_dim * i + 6);
      __m256d x31 = _mm256_broadcast_sd(ydata + n_dim * i + 7);

      // Diagonal block
      int j = i;
      __m256d y00 = _mm256_i64gather_pd(ydata + n_dim * j, index, 8);
      __m256d y01 = _mm256_i64gather_pd(ydata + n_dim * j + 1, index, 8);

      __m256d diff000 = _mm256_sub_pd(x00, y00);
      __m256d diff001 = _mm256_sub_pd(x01, y01);

      __m256d prod00 = _mm256_mul_pd(diff000, diff000);
      __m256d dists00 = _mm256_fmadd_pd(diff001, diff001, prod00);
      __m256d qnum00 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists00));
      qnum00 = _mm256_blend_pd(qnum00, zero_vec, 0b0001);
      sum = _mm256_fmadd_pd(qnum00, onehalf_vec, sum);
      _mm256_storeu_pd(q_numdata + n * i + j, qnum00);

      __m256d diff100 = _mm256_sub_pd(x10, y00);
      __m256d diff101 = _mm256_sub_pd(x11, y01);

      __m256d prod10 = _mm256_mul_pd(diff100, diff100);
      __m256d dists10 = _mm256_fmadd_pd(diff101, diff101, prod10);
      __m256d qnum10 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists10));
      qnum10 = _mm256_blend_pd(qnum10, zero_vec, 0b0010);
      sum = _mm256_fmadd_pd(qnum10, onehalf_vec, sum);
      _mm256_storeu_pd(q_numdata + n * i + n + j, qnum10);

      __m256d diff200 = _mm256_sub_pd(x20, y00);
      __m256d diff201 = _mm256_sub_pd(x21, y01);

      __m256d prod20 = _mm256_mul_pd(diff200, diff200);
      __m256d dists20 = _mm256_fmadd_pd(diff201, diff201, prod20);
      __m256d qnum20 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists20));
      qnum20 = _mm256_blend_pd(qnum20, zero_vec, 0b0100);
      sum = _mm256_fmadd_pd(qnum20, onehalf_vec, sum);
      _mm256_storeu_pd(q_numdata + n * i + 2 * n + j, qnum20);

      __m256d diff300 = _mm256_sub_pd(x30, y00);
      __m256d diff301 = _mm256_sub_pd(x31, y01);

      __m256d prod30 = _mm256_mul_pd(diff300, diff300);
      __m256d dists30 = _mm256_fmadd_pd(diff301, diff301, prod30);
      __m256d qnum30 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists30));
      qnum30 = _mm256_blend_pd(qnum30, zero_vec, 0b1000);
      sum = _mm256_fmadd_pd(qnum30, onehalf_vec, sum);
      _mm256_storeu_pd(q_numdata + n * i + 3 * n + j, qnum30);

      // Non-diagonal blocks
      j = i + 4;
      for (; j < 4 * (n / 4); j += 4) {
        y00 = _mm256_i64gather_pd(ydata + n_dim * j, index, 8);
        y01 = _mm256_i64gather_pd(ydata + n_dim * j + 1, index, 8);

        diff000 = _mm256_sub_pd(x00, y00);
        diff001 = _mm256_sub_pd(x01, y01);

        prod00 = _mm256_mul_pd(diff000, diff000);
        dists00 = _mm256_fmadd_pd(diff001, diff001, prod00);
        qnum00 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists00));
        sum = _mm256_add_pd(qnum00, sum);
        _mm256_storeu_pd(q_numdata + n * i + j, qnum00);

        diff100 = _mm256_sub_pd(x10, y00);
        diff101 = _mm256_sub_pd(x11, y01);

        prod10 = _mm256_mul_pd(diff100, diff100);
        dists10 = _mm256_fmadd_pd(diff101, diff101, prod10);
        qnum10 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists10));
        sum = _mm256_add_pd(qnum10, sum);
        _mm256_storeu_pd(q_numdata + n * i + n + j, qnum10);

        diff200 = _mm256_sub_pd(x20, y00);
        diff201 = _mm256_sub_pd(x21, y01);

        prod20 = _mm256_mul_pd(diff200, diff200);
        dists20 = _mm256_fmadd_pd(diff201, diff201, prod20);
        qnum20 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists20));
        sum = _mm256_add_pd(qnum20, sum);
        _mm256_storeu_pd(q_numdata + n * i + 2 * n + j, qnum20);

        diff300 = _mm256_sub_pd(x30, y00);
        diff301 = _mm256_sub_pd(x31, y01);

        prod30 = _mm256_mul_pd(diff300, diff300);
        dists30 = _mm256_fmadd_pd(diff301, diff301, prod30);
        qnum30 = _mm256_div_pd(one_vec, _mm256_add_pd(one_vec, dists30));
        sum = _mm256_add_pd(qnum30, sum);
        _mm256_storeu_pd(q_numdata + n * i + 3 * n + j, qnum30);

        // Fill lower triangular 4x4 block
        __m256d qnum00t, qnum10t, qnum20t, qnum30t;
        TRANSPOSE_4X4(qnum00, qnum10, qnum20, qnum30, qnum00t, qnum10t, qnum20t,
                      qnum30t);
        _mm256_storeu_pd(q_numdata + n * j + i, qnum00t);
        _mm256_storeu_pd(q_numdata + n * j + n + i, qnum10t);
        _mm256_storeu_pd(q_numdata + n * j + 2 * n + i, qnum20t);
        _mm256_storeu_pd(q_numdata + n * j + 3 * n + i, qnum30t);
      }
    }

    sum = _mm256_mul_pd(sum,
                        two_vec);  // Only upper triangular elements were summed

    // Determine normalization factor
    norm = _mm256_hadd_pd(sum, sum);
    norm = _mm256_add_pd(norm, _mm256_permute4x64_pd(norm, 0b01001110));
    norm = _mm256_div_pd(one_vec, norm);
  }

  // calculate gradient with respect to embeddings Y
  int twoi = 0;
  __m256d zero = {0, 0, 0, 0};
  for (int i = 0; i < n; i += 4) {
    int twoj = 0;
    __m256d yleft, yright, y1, y2, yfixleft1, yfixleft2, yfixleft3, yfixleft4,
        yfixright1, yfixright2, yfixright3, yfixright4;
    __m256d valueleft1 = _mm256_setzero_pd();
    __m256d valueright1 = _mm256_setzero_pd();
    __m256d valueleft2 = _mm256_setzero_pd();
    __m256d valueright2 = _mm256_setzero_pd();
    __m256d valueleft3 = _mm256_setzero_pd();
    __m256d valueright3 = _mm256_setzero_pd();
    __m256d valueleft4 = _mm256_setzero_pd();
    __m256d valueright4 = _mm256_setzero_pd();
    yfixleft1 = _mm256_broadcast_sd(ydata + twoi);
    yfixleft2 = _mm256_broadcast_sd(ydata + twoi + 2);
    yfixleft3 = _mm256_broadcast_sd(ydata + twoi + 4);
    yfixleft4 = _mm256_broadcast_sd(ydata + twoi + 6);
    yfixright1 = _mm256_broadcast_sd(ydata + twoi + 1);
    yfixright2 = _mm256_broadcast_sd(ydata + twoi + 3);
    yfixright3 = _mm256_broadcast_sd(ydata + twoi + 5);
    yfixright4 = _mm256_broadcast_sd(ydata + twoi + 7);
    for (int j = 0; j < n; j += 4) {
      __m256d p1, p2, p3, p4, q1, q2, q3, q4, qnum1, qnum2, qnum3, qnum4, tmp1,
          tmp2, tmp3, tmp4;
      y1 = _mm256_load_pd(ydata + twoj);
      y2 = _mm256_load_pd(ydata + twoj + 4);
      // sort such that we have column wise 4 y elements
      yleft = _mm256_unpacklo_pd(y1, y2);
      yright = _mm256_unpackhi_pd(y1, y2);
      yleft = _mm256_permute4x64_pd(yleft, ymask);
      yright = _mm256_permute4x64_pd(yright, ymask);
      p1 = _mm256_load_pd(pdata + i * n + j);
      p2 = _mm256_load_pd(pdata + i * n + j + n);
      p3 = _mm256_load_pd(pdata + i * n + j + 2 * n);
      p4 = _mm256_load_pd(pdata + i * n + j + 3 * n);
      qnum1 = _mm256_load_pd(q_numdata + i * n + j);
      qnum2 = _mm256_load_pd(q_numdata + i * n + j + n);
      qnum3 = _mm256_load_pd(q_numdata + i * n + j + 2 * n);
      qnum4 = _mm256_load_pd(q_numdata + i * n + j + 3 * n);
      q1 = _mm256_mul_pd(norm, qnum1);
      q2 = _mm256_mul_pd(norm, qnum2);
      q3 = _mm256_mul_pd(norm, qnum3);
      q4 = _mm256_mul_pd(norm, qnum4);

      // Set minimum probability
      __m256d vec_min_prob = _mm256_set1_pd(kMinimumProbability);
      q1 = _mm256_max_pd(q1, vec_min_prob);
      q2 = _mm256_max_pd(q2, vec_min_prob);
      q3 = _mm256_max_pd(q3, vec_min_prob);
      q4 = _mm256_max_pd(q4, vec_min_prob);

      tmp1 = _mm256_mul_pd(_mm256_sub_pd(p1, q1), qnum1);
      tmp2 = _mm256_mul_pd(_mm256_sub_pd(p2, q2), qnum2);
      tmp3 = _mm256_mul_pd(_mm256_sub_pd(p3, q3), qnum3);
      tmp4 = _mm256_mul_pd(_mm256_sub_pd(p4, q4), qnum4);
      valueleft1 =
          _mm256_fmadd_pd(tmp1, _mm256_sub_pd(yfixleft1, yleft), valueleft1);
      valueleft2 =
          _mm256_fmadd_pd(tmp2, _mm256_sub_pd(yfixleft2, yleft), valueleft2);
      valueleft3 =
          _mm256_fmadd_pd(tmp3, _mm256_sub_pd(yfixleft3, yleft), valueleft3);
      valueleft4 =
          _mm256_fmadd_pd(tmp4, _mm256_sub_pd(yfixleft4, yleft), valueleft4);
      valueright1 =
          _mm256_fmadd_pd(tmp1, _mm256_sub_pd(yfixright1, yright), valueright1);
      valueright2 =
          _mm256_fmadd_pd(tmp2, _mm256_sub_pd(yfixright2, yright), valueright2);
      valueright3 =
          _mm256_fmadd_pd(tmp3, _mm256_sub_pd(yfixright3, yright), valueright3);
      valueright4 =
          _mm256_fmadd_pd(tmp4, _mm256_sub_pd(yfixright4, yright), valueright4);
      twoj += 8;
    }

    double *v = (double *)aligned_alloc(32, 16 * sizeof(double));
    _mm256_store_pd(v, _mm256_hadd_pd(valueleft1, valueright1));
    _mm256_store_pd(v + 4, _mm256_hadd_pd(valueleft2, valueright2));
    _mm256_store_pd(v + 8, _mm256_hadd_pd(valueleft3, valueright3));
    _mm256_store_pd(v + 12, _mm256_hadd_pd(valueleft4, valueright4));
    __m256d values_left, values_right;
    values_left = _mm256_set_pd(v[12] + v[14], v[8] + v[10], v[4] + v[6],
                                v[0] + v[2]);  // correct
    values_right = _mm256_set_pd(v[13] + v[15], v[9] + v[11], v[5] + v[7],
                                 v[1] + v[3]);  // correct

    __m256d ydeltaleft, ydeltaright, ydelta1, ydelta2, pos_grad_left,
        pos_grad_right, pos_delta_left, pos_delta_right, gainsleft, gainsright,
        gains0, gains1;
    ydelta1 = _mm256_load_pd(ydeltadata + twoi);
    ydelta2 = _mm256_load_pd(ydeltadata + twoi + 4);
    // sort such that we have column wise 4 ydelta elements
    ydeltaleft = _mm256_unpacklo_pd(ydelta1, ydelta2);
    ydeltaright = _mm256_unpackhi_pd(ydelta1, ydelta2);
    ydeltaleft = _mm256_permute4x64_pd(ydeltaleft, ymask);
    ydeltaright = _mm256_permute4x64_pd(ydeltaright, ymask);

    // compute boolean masks
    pos_grad_left = _mm256_cmp_pd(values_left, zero, _CMP_GT_OQ);
    pos_grad_right = _mm256_cmp_pd(values_right, zero, _CMP_GT_OQ);
    pos_delta_left = _mm256_cmp_pd(ydeltaleft, zero, _CMP_GT_OQ);
    pos_delta_right = _mm256_cmp_pd(ydeltaright, zero, _CMP_GT_OQ);

    // load gains
    gains0 = _mm256_load_pd(gainsdata + twoi);
    gains1 = _mm256_load_pd(gainsdata + twoi + 4);

    // sort gains into left and right
    gainsleft = _mm256_unpacklo_pd(gains0, gains1);
    gainsright = _mm256_unpackhi_pd(gains0, gains1);
    gainsleft = _mm256_permute4x64_pd(gainsleft, ymask);
    gainsright = _mm256_permute4x64_pd(gainsright, ymask);

    __m256d gainsmul_left, gainsmul_right, gainsplus_left, gainsplus_right,
        mask_left, mask_right;
    __m256d mulconst = {0.8, 0.8, 0.8, 0.8};
    __m256d addconst = {0.2, 0.2, 0.2, 0.2};
    gainsmul_left = _mm256_mul_pd(gainsleft, mulconst);
    gainsmul_right = _mm256_mul_pd(gainsright, mulconst);
    gainsplus_left = _mm256_add_pd(gainsleft, addconst);
    gainsplus_right = _mm256_add_pd(gainsright, addconst);
    mask_left = _mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_mm256_castpd_si256(pos_grad_left),
                           _mm256_castpd_si256(pos_delta_left)));
    mask_right = _mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_mm256_castpd_si256(pos_grad_right),
                           _mm256_castpd_si256(pos_delta_right)));

    gainsmul_left = _mm256_and_pd(mask_left, gainsmul_left);
    gainsmul_right = _mm256_and_pd(mask_right, gainsmul_right);
    gainsplus_left = _mm256_andnot_pd(mask_left, gainsplus_left);
    gainsplus_right = _mm256_andnot_pd(mask_right, gainsplus_right);

    gainsleft = _mm256_or_pd(gainsmul_left, gainsplus_left);
    gainsright = _mm256_or_pd(gainsmul_right, gainsplus_right);

    __m256d kmask_left, kmask_right;
    __m256d kmin = {kMinGain, kMinGain, kMinGain, kMinGain};
    kmask_left = _mm256_cmp_pd(gainsleft, kmin, _CMP_LT_OQ);
    kmask_right = _mm256_cmp_pd(gainsright, kmin, _CMP_LT_OQ);
    gainsleft = _mm256_blendv_pd(gainsleft, kmin, kmask_left);
    gainsright = _mm256_blendv_pd(gainsright, kmin, kmask_right);

    // unsort again
    gains0 = _mm256_permute4x64_pd(gainsleft, ymask);
    gains1 = _mm256_permute4x64_pd(gainsright, ymask);

    _mm256_store_pd(gainsdata + twoi, _mm256_unpacklo_pd(gains0, gains1));
    _mm256_store_pd(gainsdata + twoi + 4, _mm256_unpackhi_pd(gains0, gains1));

    __m256d momentum_v = {momentum, momentum, momentum, momentum};
    __m256d fourketa = {fourkEta, fourkEta, fourkEta, fourkEta};
    __m256d fourketa_values_left = _mm256_mul_pd(fourketa, values_left);
    __m256d fourketa_values_right = _mm256_mul_pd(fourketa, values_right);
    gainsleft = _mm256_mul_pd(gainsleft, fourketa_values_left);
    gainsright = _mm256_mul_pd(gainsright, fourketa_values_right);
    ydeltaleft = _mm256_fmsub_pd(momentum_v, ydeltaleft, gainsleft);
    ydeltaright = _mm256_fmsub_pd(momentum_v, ydeltaright, gainsright);

    ydelta1 = _mm256_permute4x64_pd(ydeltaleft, ymask);
    ydelta2 = _mm256_permute4x64_pd(ydeltaright, ymask);

    _mm256_store_pd(ydeltadata + twoi, _mm256_unpacklo_pd(ydelta1, ydelta2));
    _mm256_store_pd(ydeltadata + twoi + 4,
                    _mm256_unpackhi_pd(ydelta1, ydelta2));

    twoi += 8;
  }

  __m256d mean = {0, 0, 0, 0};
  __m256d n_vec = _mm256_set1_pd(n);
  __m256d y, ydelta, mean_shuffled, mean_shuffled2;
  const int mean_mask =
      0b10001101;  // setup such that we have [mean1 mean3 mean0 mean2]
  const int mean_mask2 =
      0b11011000;  // setup such that we have [mean0 mean2 mean1 mean3]
  const int mean_last_mask = 0b0110;
  int twon = 2 * n;
  for (int i = 0; i < twon; i += 4) {
    // load y, ydelta
    y = _mm256_load_pd(ydata + i);
    ydelta = _mm256_load_pd(ydeltadata + i);

    // update step
    y = _mm256_add_pd(y, ydelta);
    mean = _mm256_add_pd(mean, y);

    _mm256_store_pd(ydata + i, y);
  }
  mean_shuffled = _mm256_permute4x64_pd(mean, mean_mask);
  mean_shuffled2 = _mm256_permute4x64_pd(mean, mean_mask2);
  mean = _mm256_hadd_pd(
      mean_shuffled2,
      mean_shuffled);  // mean now holds [mean0 mean1 mean1 mean0]
  mean = _mm256_permute_pd(mean, mean_last_mask);
  // take mean
  mean = _mm256_div_pd(mean, n_vec);
  // center
  for (int i = 0; i < twon; i += 4) {
    y = _mm256_load_pd(ydata + i);
    y = _mm256_sub_pd(y, mean);
    _mm256_store_pd(ydata + i, y);
  }
}

void tsne_vec3(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim) {
  int n = X->nrows;

  // compute high level joint probabilities
  // _joint_probs_vec(X, &var->P, &var->D);
  _joint_probs_vec(X, &var->P, &var->D);

  // determine embeddings
  // initialisations
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      var->gains.data[i * n_dim + j] = 1;
    }
  }

  double momentum = kInitialMomentum;
  for (int iter = 0; iter < kGradDescMaxIter; iter++) {
    // early exaggeration only for first 100 iterations
    if (iter == 100) {
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double value = var->P.data[i * n + j] / 4;
          var->P.data[i * n + j] = value;
          var->P.data[j * n + i] = value;
        }
      }
    }

    // reduce momentum at iteration 20
    if (iter == 20) momentum = kFinalMomentum;

    _grad_desc_vec3(Y, var, n, n_dim, momentum);
  }
}
