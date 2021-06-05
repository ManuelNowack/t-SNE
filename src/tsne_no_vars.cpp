#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

#define WRITE_ALL_VARS 1

void grad_desc_no_vars_baseline(Matrix *Y, tsne_var_t *var, int n, int m,
                                double momentum) {
  // START: Euclidean Distances
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y->data[i * m + k] - Y->data[j * m + k];
        sum += dist * dist;
      }
      var->D.data[i * n + j] = sum;
      var->D.data[j * n + i] = sum;
    }
  }

  // set diagonal entries
  for (int i = 0; i < n; i++) {
    var->D.data[i * n + i] = 0.0;
  }
  // END: Euclidean Distances

  // START: Affinities
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      const double value = 1.0 / (1 + var->D.data[i * n + j]);
      var->Q_numerators.data[i * n + j] = value;
      var->Q_numerators.data[j * n + i] = value;
      sum += value;
    }
  }

  // set diagonal elements
  for (int i = 0; i < n; i++) {
    var->Q.data[i * n + i] = 0.0;
  }

  const double norm = 0.5 / sum;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = var->Q_numerators.data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      var->Q.data[i * n + j] = value;
      var->Q.data[j * n + i] = value;
    }
  }
  // END: Affinities

  // START: Gradient Descent
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      const double value = (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
                           var->Q_numerators.data[i * n + j];
      var->tmp.data[i * n + j] = value;
      var->tmp.data[j * n + i] = value;
    }
    var->tmp.data[i * n + i] = 0.0;
  }
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      double value = 0.0;
      for (int j = 0; j < n; j++) {
        value += var->tmp.data[i * n + j] *
                 (Y->data[i * m + k] - Y->data[j * m + k]);
      }
      value *= 4.0;
      var->grad_Y.data[i * m + k] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (var->grad_Y.data[i * m + j] > 0);
      const bool positive_delta = (var->Y_delta.data[i * m + j] > 0);
      double value = var->gains.data[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      var->gains.data[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value =
          momentum * var->Y_delta.data[i * m + j] -
          kEta * var->gains.data[i * m + j] * var->grad_Y.data[i * m + j];
      var->Y_delta.data[i * m + j] = value;
      Y->data[i * m + j] += value;
    }
  }

  // center each dimension at 0
  double means[m];
  for (int j = 0; j < m; j++) {
    means[j] = 0.0;
  }
  // accumulate
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      means[j] += Y->data[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y->data[i * m + j] -= means[j];
    }
  }
  // END: Gradient Descent
}

void grad_desc_no_vars_tmp(Matrix *Y, tsne_var_t *var, int n, int m,
                           double momentum) {
  // START: Euclidean Distances
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y->data[i * m + k] - Y->data[j * m + k];
        sum += dist * dist;
      }
      var->D.data[i * n + j] = sum;
      var->D.data[j * n + i] = sum;
    }
  }

  // set diagonal entries
  for (int i = 0; i < n; i++) {
    var->D.data[i * n + i] = 0.0;
  }
  // END: Euclidean Distances

  // START: Affinities
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      const double value = 1.0 / (1 + var->D.data[i * n + j]);
      var->Q_numerators.data[i * n + j] = value;
      var->Q_numerators.data[j * n + i] = value;
      sum += value;
    }
  }

  // set diagonal elements
  for (int i = 0; i < n; i++) {
    var->Q.data[i * n + i] = 0.0;
  }

  const double norm = 0.5 / sum;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = var->Q_numerators.data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      var->Q.data[i * n + j] = value;
      var->Q.data[j * n + i] = value;
    }
  }
  // END: Affinities

  // START: Gradient Descent
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        const double tmp_value =
            (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
            var->Q_numerators.data[i * n + j];
        const double value =
            tmp_value * (Y->data[i * m + k] - Y->data[j * m + k]);
        sum += value;
        if (WRITE_ALL_VARS) {
          var->tmp.data[i * n + j] = tmp_value;
        }
      }
      const double value = 4.0 * sum;
      var->grad_Y.data[i * m + k] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (var->grad_Y.data[i * m + j] > 0);
      const bool positive_delta = (var->Y_delta.data[i * m + j] > 0);
      double value = var->gains.data[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      var->gains.data[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value =
          momentum * var->Y_delta.data[i * m + j] -
          kEta * var->gains.data[i * m + j] * var->grad_Y.data[i * m + j];
      var->Y_delta.data[i * m + j] = value;
      Y->data[i * m + j] += value;
    }
  }

  // center each dimension at 0
  double means[m];
  for (int j = 0; j < m; j++) {
    means[j] = 0.0;
  }
  // accumulate
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      means[j] += Y->data[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y->data[i * m + j] -= means[j];
    }
  }
  // END: Gradient Descent
}

void grad_desc_no_vars_D(Matrix *Y, tsne_var_t *var, int n, int m,
                         double momentum) {
  // START: Affinities
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y->data[i * m + k] - Y->data[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      var->Q_numerators.data[i * n + j] = value;
      var->Q_numerators.data[j * n + i] = value;
      sum += value;
      if (WRITE_ALL_VARS) {
        var->D.data[i * n + j] = dist_sum;
        var->D.data[j * n + i] = dist_sum;
      }
    }
  }

  // set diagonal elements
  for (int i = 0; i < n; i++) {
    var->Q.data[i * n + i] = 0.0;
    if (WRITE_ALL_VARS) {
      var->D.data[i * n + i] = 0.0;
    }
  }

  const double norm = 0.5 / sum;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = var->Q_numerators.data[i * n + j];
      value *= norm;
      if (value < kMinimumProbability) {
        value = kMinimumProbability;
      }
      var->Q.data[i * n + j] = value;
      var->Q.data[j * n + i] = value;
    }
  }
  // END: Affinities

  // START: Gradient Descent
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        const double tmp_value =
            (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
            var->Q_numerators.data[i * n + j];
        const double value =
            tmp_value * (Y->data[i * m + k] - Y->data[j * m + k]);
        sum += value;
        if (WRITE_ALL_VARS) {
          var->tmp.data[i * n + j] = tmp_value;
        }
      }
      const double value = 4.0 * sum;
      var->grad_Y.data[i * m + k] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (var->grad_Y.data[i * m + j] > 0);
      const bool positive_delta = (var->Y_delta.data[i * m + j] > 0);
      double value = var->gains.data[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      var->gains.data[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value =
          momentum * var->Y_delta.data[i * m + j] -
          kEta * var->gains.data[i * m + j] * var->grad_Y.data[i * m + j];
      var->Y_delta.data[i * m + j] = value;
      Y->data[i * m + j] += value;
    }
  }

  // center each dimension at 0
  double means[m];
  for (int j = 0; j < m; j++) {
    means[j] = 0.0;
  }
  // accumulate
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      means[j] += Y->data[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y->data[i * m + j] -= means[j];
    }
  }
  // END: Gradient Descent
}

void grad_desc_no_vars_Q(Matrix *Y, tsne_var_t *var, int n, int m,
                         double momentum) {
  // START: Affinities
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y->data[i * m + k] - Y->data[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      var->Q_numerators.data[i * n + j] = value;
      var->Q_numerators.data[j * n + i] = value;
      sum += value;
      if (WRITE_ALL_VARS) {
        var->D.data[i * n + j] = dist_sum;
        var->D.data[j * n + i] = dist_sum;
      }
    }
  }

  // set diagonal elements
  for (int i = 0; i < n; i++) {
    var->Q.data[i * n + i] = 0.0;
    if (WRITE_ALL_VARS) {
      var->D.data[i * n + i] = 0.0;
    }
  }

  const double norm = 0.5 / sum;
  // END: Affinities

  // START: Gradient Descent
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        double q_value = var->Q_numerators.data[i * n + j];
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (var->P.data[i * n + j] - q_value) *
                                 var->Q_numerators.data[i * n + j];
        const double value =
            tmp_value * (Y->data[i * m + k] - Y->data[j * m + k]);
        sum += value;
        if (WRITE_ALL_VARS) {
          var->tmp.data[i * n + j] = tmp_value;
          if (i > j) {
            var->Q.data[i * n + j] = q_value;
            var->Q.data[j * n + i] = q_value;
          }
        }
      }
      const double value = 4.0 * sum;
      var->grad_Y.data[i * m + k] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (var->grad_Y.data[i * m + j] > 0);
      const bool positive_delta = (var->Y_delta.data[i * m + j] > 0);
      double value = var->gains.data[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      var->gains.data[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value =
          momentum * var->Y_delta.data[i * m + j] -
          kEta * var->gains.data[i * m + j] * var->grad_Y.data[i * m + j];
      var->Y_delta.data[i * m + j] = value;
      Y->data[i * m + j] += value;
    }
  }

  // center each dimension at 0
  double means[m];
  for (int j = 0; j < m; j++) {
    means[j] = 0.0;
  }
  // accumulate
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      means[j] += Y->data[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y->data[i * m + j] -= means[j];
    }
  }
  // END: Gradient Descent
}

void tsne_no_vars(Matrix *X, Matrix *Y, tsne_var_t *var, int m) {
  int n = X->nrows;

  joint_probs_baseline(X, &var->P, &var->D);

  // determine embeddings
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      var->Y_delta.data[i * m + j] = 0.0;
      var->gains.data[i * m + j] = 1.0;
    }
  }

  double momentum = kInitialMomentum;
  for (int iter = 0; iter < kGradDescMaxIter; iter++) {
    // early exaggeration only for first 100 iterations
    if (iter == 100) {
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double value = var->P.data[i * n + j] / 4.0;
          var->P.data[i * n + j] = value;
          var->P.data[j * n + i] = value;
        }
      }
    }

    // reduce momentum at iteration 20
    if (iter == 20) {
      momentum = kFinalMomentum;
    }

    grad_desc_no_vars_baseline(Y, var, n, m, momentum);
  }
}
