#include <immintrin.h>
#include <math.h>
#include <tsne/debug.h>
#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>
#include <vectorclass/vectormath_exp.h>

void joint_probs_unroll8(Matrix *X, Matrix *P, Matrix *D) {
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
    for (int i = 0; i < n; i++) {
      probabilities[i] = probabilities[i] / normalizer;
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

void joint_probs_avx_fma_acc4(Matrix *X, Matrix *P, Matrix *D) {
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
    for (int i = 0; i < n; i++) {
      probabilities[i] = probabilities[i] / normalizer;
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
