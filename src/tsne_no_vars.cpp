#include <assert.h>
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

#define WRITE_ALL_VARS 0

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
    for (int l = 0; l < m; l++) {
      double value = 0.0;
      for (int j = 0; j < n; j++) {
        value += var->tmp.data[i * n + j] *
                 (Y->data[i * m + l] - Y->data[j * m + l]);
      }
      value *= 4.0;
      var->grad_Y.data[i * m + l] = value;
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
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        const double tmp_value =
            (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
            var->Q_numerators.data[i * n + j];
        const double value =
            tmp_value * (Y->data[i * m + l] - Y->data[j * m + l]);
        sum += value;
        if (WRITE_ALL_VARS) {
          var->tmp.data[i * n + j] = tmp_value;
        }
      }
      const double value = 4.0 * sum;
      var->grad_Y.data[i * m + l] = value;
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
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        const double tmp_value =
            (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
            var->Q_numerators.data[i * n + j];
        const double value =
            tmp_value * (Y->data[i * m + l] - Y->data[j * m + l]);
        sum += value;
        if (WRITE_ALL_VARS) {
          var->tmp.data[i * n + j] = tmp_value;
        }
      }
      const double value = 4.0 * sum;
      var->grad_Y.data[i * m + l] = value;
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
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        double q_value = var->Q_numerators.data[i * n + j];
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (var->P.data[i * n + j] - q_value) *
                                 var->Q_numerators.data[i * n + j];
        const double value =
            tmp_value * (Y->data[i * m + l] - Y->data[j * m + l]);
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
      var->grad_Y.data[i * m + l] = value;
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

void grad_desc_no_vars_Q_numerators(Matrix *Y, tsne_var_t *var, int n, int m,
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
      sum += value;
      if (WRITE_ALL_VARS) {
        var->Q_numerators.data[i * n + j] = value;
        var->Q_numerators.data[j * n + i] = value;
        var->D.data[i * n + j] = dist_sum;
        var->D.data[j * n + i] = dist_sum;
      }
    }
  }

  // set diagonal elements
  if (WRITE_ALL_VARS) {
    for (int i = 0; i < n; i++) {
      var->Q.data[i * n + i] = 0.0;
      var->D.data[i * n + i] = 0.0;
    }
  }

  const double norm = 0.5 / sum;
  // END: Affinities

  // START: Gradient Descent
  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y->data[i * m + k] - Y->data[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value =
            (i == j) ? 0.0
                     : (var->P.data[i * n + j] - q_value) * q_numerator_value;
        const double value =
            tmp_value * (Y->data[i * m + l] - Y->data[j * m + l]);
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
      var->grad_Y.data[i * m + l] = value;
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

void grad_desc_no_vars_scalar_pure(double *Y, const double *P, double *grad_Y,
                                   double *Y_delta, double *gains, int n, int m,
                                   double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value =
            (i == j) ? 0.0 : (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double value = 4.0 * sum;
      grad_Y[i * m + l] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_scalar(Matrix *Y, tsne_var_t *var, int n, int m,
                              double momentum) {
  grad_desc_no_vars_scalar_pure(Y->data, var->P.data, var->grad_Y.data,
                                var->Y_delta.data, var->gains.data, n, m,
                                momentum);
}

void grad_desc_no_vars_no_if_pure(double *Y, const double *P, double *grad_Y,
                                  double *Y_delta, double *gains, int n, int m,
                                  double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double value = 4.0 * sum;
      grad_Y[i * m + l] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_no_if(Matrix *Y, tsne_var_t *var, int n, int m,
                             double momentum) {
  grad_desc_no_vars_no_if_pure(Y->data, var->P.data, var->grad_Y.data,
                               var->Y_delta.data, var->gains.data, n, m,
                               momentum);
}

void grad_desc_no_vars_grad_pure(double *Y, const double *P, double *Y_delta,
                                 double *gains, int n, int m, double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double grad = 4.0 * sum;
      const double old_delta = Y_delta[i * m + l];
      const bool positive_grad = (grad > 0);
      const bool positive_delta = (old_delta > 0);
      double gain = gains[i * m + l];
      if (positive_grad == positive_delta) {
        gain *= 0.8;
      } else {
        gain += 0.2;
      }
      if (gain < kMinGain) {
        gain = kMinGain;
      }
      gains[i * m + l] = gain;
      const double new_delta = momentum * old_delta - kEta * gain * grad;
      Y_delta[i * m + l] = new_delta;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] += Y_delta[i * m + j];
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_grad(Matrix *Y, tsne_var_t *var, int n, int m,
                             double momentum) {
  grad_desc_no_vars_grad_pure(Y->data, var->P.data, var->Y_delta.data,
                              var->gains.data, n, m, momentum);
}

void grad_desc_no_vars_means_pure(double *Y, const double *P, double *Y_delta,
                                  double *gains, int n, int m, double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  double means[m];
  for (int j = 0; j < m; j++) {
    means[j] = 0.0;
  }
  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double grad = 4.0 * sum;
      const double old_delta = Y_delta[i * m + l];
      const bool positive_grad = (grad > 0);
      const bool positive_delta = (old_delta > 0);
      double gain = gains[i * m + l];
      if (positive_grad == positive_delta) {
        gain *= 0.8;
      } else {
        gain += 0.2;
      }
      if (gain < kMinGain) {
        gain = kMinGain;
      }
      gains[i * m + l] = gain;
      const double new_delta = momentum * old_delta - kEta * gain * grad;
      Y_delta[i * m + l] = new_delta;
      means[l] += Y[i * m + l] + new_delta;
    }
  }

  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] += Y_delta[i * m + j];
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_means(Matrix *Y, tsne_var_t *var, int n, int m,
                             double momentum) {
  grad_desc_no_vars_means_pure(Y->data, var->P.data, var->Y_delta.data,
                              var->gains.data, n, m, momentum);
}

void grad_desc_no_vars_unroll2_pure(double *Y, const double *P, double *grad_Y,
                                    double *Y_delta, double *gains, int n,
                                    int m, double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    constexpr int unroll_factor = 2;
    const int begin = (i + 4) / 4 * 4;  // first 32-byte aligned address after i
    const int end = begin + (n - begin) / unroll_factor * unroll_factor;
    // front
    for (int j = i + 1; j < begin; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
    // main
    for (int j = begin; j < end; j += unroll_factor) {
      double dist_sum_1 = 0.0;
      double dist_sum_2 = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist_1 = Y[i * m + k] - Y[j * m + k];
        const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
        dist_sum_1 += dist_1 * dist_1;
        dist_sum_2 += dist_2 * dist_2;
      }
      const double value_1 = 1.0 / (1.0 + dist_sum_1);
      const double value_2 = 1.0 / (1.0 + dist_sum_2);
      sum += value_1;
      sum += value_2;
    }
    // back
    for (int j = end; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      constexpr int unroll_factor = 2;
      const int end = n / unroll_factor * unroll_factor;
      sum = 0.0;
      // main
      for (int j = 0; j < end; j += unroll_factor) {
        double dist_sum_1 = 0.0;
        double dist_sum_2 = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist_1 = Y[i * m + k] - Y[j * m + k];
          const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
          dist_sum_1 += dist_1 * dist_1;
          dist_sum_2 += dist_2 * dist_2;
        }
        const double q_numerator_value_1 = 1.0 / (1.0 + dist_sum_1);
        const double q_numerator_value_2 = 1.0 / (1.0 + dist_sum_2);

        double q_value_1 = q_numerator_value_1;
        q_value_1 *= norm;
        if (q_value_1 < kMinimumProbability) {
          q_value_1 = kMinimumProbability;
        }
        double q_value_2 = q_numerator_value_2;
        q_value_2 *= norm;
        if (q_value_2 < kMinimumProbability) {
          q_value_2 = kMinimumProbability;
        }

        const double tmp_value_1 = (P[i * n + j] - q_value_1) * q_numerator_value_1;
        const double tmp_value_2 = (P[i * n + (j + 1)] - q_value_2) * q_numerator_value_2;
        const double value_1 = tmp_value_1 * (Y[i * m + l] - Y[j * m + l]);
        const double value_2 = tmp_value_2 * (Y[i * m + l] - Y[(j + 1) * m + l]);
        sum += value_1;
        sum += value_2;
      }
      // end
      for (int j = end; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double value = 4.0 * sum;
      grad_Y[i * m + l] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_unroll2(Matrix *Y, tsne_var_t *var, int n, int m,
                               double momentum) {
  grad_desc_no_vars_unroll2_pure(Y->data, var->P.data, var->grad_Y.data,
                                 var->Y_delta.data, var->gains.data, n, m,
                                 momentum);
}

void grad_desc_no_vars_unroll4_pure(double *Y, const double *P, double *grad_Y,
                                    double *Y_delta, double *gains, int n,
                                    int m, double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    constexpr int unroll_factor = 4;
    const int begin = (i + 4) / 4 * 4;  // first 32-byte aligned address after i
    const int end = begin + (n - begin) / unroll_factor * unroll_factor;
    // front
    for (int j = i + 1; j < begin; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
    // main
    for (int j = begin; j < end; j += unroll_factor) {
      double dist_sum_1 = 0.0;
      double dist_sum_2 = 0.0;
      double dist_sum_3 = 0.0;
      double dist_sum_4 = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist_1 = Y[i * m + k] - Y[j * m + k];
        const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
        const double dist_3 = Y[i * m + k] - Y[(j + 2) * m + k];
        const double dist_4 = Y[i * m + k] - Y[(j + 3) * m + k];
        dist_sum_1 += dist_1 * dist_1;
        dist_sum_2 += dist_2 * dist_2;
        dist_sum_3 += dist_3 * dist_3;
        dist_sum_4 += dist_4 * dist_4;
      }
      const double value_1 = 1.0 / (1.0 + dist_sum_1);
      const double value_2 = 1.0 / (1.0 + dist_sum_2);
      const double value_3 = 1.0 / (1.0 + dist_sum_3);
      const double value_4 = 1.0 / (1.0 + dist_sum_4);
      sum += value_1;
      sum += value_2;
      sum += value_3;
      sum += value_4;
    }
    // back
    for (int j = end; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      constexpr int unroll_factor = 4;
      const int end = n / unroll_factor * unroll_factor;
      sum = 0.0;
      // main
      for (int j = 0; j < end; j += unroll_factor) {
        double dist_sum_1 = 0.0;
        double dist_sum_2 = 0.0;
        double dist_sum_3 = 0.0;
        double dist_sum_4 = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist_1 = Y[i * m + k] - Y[j * m + k];
          const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
          const double dist_3 = Y[i * m + k] - Y[(j + 2) * m + k];
          const double dist_4 = Y[i * m + k] - Y[(j + 3) * m + k];
          dist_sum_1 += dist_1 * dist_1;
          dist_sum_2 += dist_2 * dist_2;
          dist_sum_3 += dist_3 * dist_3;
          dist_sum_4 += dist_4 * dist_4;
        }
        const double q_numerator_value_1 = 1.0 / (1.0 + dist_sum_1);
        const double q_numerator_value_2 = 1.0 / (1.0 + dist_sum_2);
        const double q_numerator_value_3 = 1.0 / (1.0 + dist_sum_3);
        const double q_numerator_value_4 = 1.0 / (1.0 + dist_sum_4);

        double q_value_1 = q_numerator_value_1;
        q_value_1 *= norm;
        if (q_value_1 < kMinimumProbability) {
          q_value_1 = kMinimumProbability;
        }
        double q_value_2 = q_numerator_value_2;
        q_value_2 *= norm;
        if (q_value_2 < kMinimumProbability) {
          q_value_2 = kMinimumProbability;
        }
        double q_value_3 = q_numerator_value_3;
        q_value_3 *= norm;
        if (q_value_3 < kMinimumProbability) {
          q_value_3 = kMinimumProbability;
        }
        double q_value_4 = q_numerator_value_4;
        q_value_4 *= norm;
        if (q_value_4 < kMinimumProbability) {
          q_value_4 = kMinimumProbability;
        }

        const double tmp_value_1 = (P[i * n + j] - q_value_1) * q_numerator_value_1;
        const double tmp_value_2 = (P[i * n + (j + 1)] - q_value_2) * q_numerator_value_2;
        const double tmp_value_3 = (P[i * n + (j + 2)] - q_value_3) * q_numerator_value_3;
        const double tmp_value_4 = (P[i * n + (j + 3)] - q_value_4) * q_numerator_value_4;
        const double value_1 = tmp_value_1 * (Y[i * m + l] - Y[j * m + l]);
        const double value_2 = tmp_value_2 * (Y[i * m + l] - Y[(j + 1) * m + l]);
        const double value_3 = tmp_value_3 * (Y[i * m + l] - Y[(j + 2) * m + l]);
        const double value_4 = tmp_value_4 * (Y[i * m + l] - Y[(j + 3) * m + l]);
        sum += value_1;
        sum += value_2;
        sum += value_3;
        sum += value_4;
      }
      // end
      for (int j = end; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double value = 4.0 * sum;
      grad_Y[i * m + l] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_unroll4(Matrix *Y, tsne_var_t *var, int n, int m,
                               double momentum) {
  grad_desc_no_vars_unroll4_pure(Y->data, var->P.data, var->grad_Y.data,
                                 var->Y_delta.data, var->gains.data, n, m,
                                 momentum);
}

void grad_desc_no_vars_unroll6_pure(double *Y, const double *P, double *grad_Y,
                                    double *Y_delta, double *gains, int n,
                                    int m, double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    constexpr int unroll_factor = 4;
    const int begin = (i + 4) / 4 * 4;  // first 32-byte aligned address after i
    const int end = begin + (n - begin) / unroll_factor * unroll_factor;
    // front
    for (int j = i + 1; j < begin; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
    // main
    for (int j = begin; j < end; j += unroll_factor) {
      double dist_sum_1 = 0.0;
      double dist_sum_2 = 0.0;
      double dist_sum_3 = 0.0;
      double dist_sum_4 = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist_1 = Y[i * m + k] - Y[j * m + k];
        const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
        const double dist_3 = Y[i * m + k] - Y[(j + 2) * m + k];
        const double dist_4 = Y[i * m + k] - Y[(j + 3) * m + k];
        dist_sum_1 += dist_1 * dist_1;
        dist_sum_2 += dist_2 * dist_2;
        dist_sum_3 += dist_3 * dist_3;
        dist_sum_4 += dist_4 * dist_4;
      }
      const double value_1 = 1.0 / (1.0 + dist_sum_1);
      const double value_2 = 1.0 / (1.0 + dist_sum_2);
      const double value_3 = 1.0 / (1.0 + dist_sum_3);
      const double value_4 = 1.0 / (1.0 + dist_sum_4);
      sum += value_1;
      sum += value_2;
      sum += value_3;
      sum += value_4;
    }
    // back
    for (int j = end; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      constexpr int unroll_factor = 6;
      const int end = n / unroll_factor * unroll_factor;
      sum = 0.0;
      // main
      for (int j = 0; j < end; j += unroll_factor) {
        double dist_sum_1 = 0.0;
        double dist_sum_2 = 0.0;
        double dist_sum_3 = 0.0;
        double dist_sum_4 = 0.0;
        double dist_sum_5 = 0.0;
        double dist_sum_6 = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist_1 = Y[i * m + k] - Y[j * m + k];
          const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
          const double dist_3 = Y[i * m + k] - Y[(j + 2) * m + k];
          const double dist_4 = Y[i * m + k] - Y[(j + 3) * m + k];
          const double dist_5 = Y[i * m + k] - Y[(j + 4) * m + k];
          const double dist_6 = Y[i * m + k] - Y[(j + 5) * m + k];
          dist_sum_1 += dist_1 * dist_1;
          dist_sum_2 += dist_2 * dist_2;
          dist_sum_3 += dist_3 * dist_3;
          dist_sum_4 += dist_4 * dist_4;
          dist_sum_5 += dist_5 * dist_5;
          dist_sum_6 += dist_6 * dist_6;
        }
        const double q_numerator_value_1 = 1.0 / (1.0 + dist_sum_1);
        const double q_numerator_value_2 = 1.0 / (1.0 + dist_sum_2);
        const double q_numerator_value_3 = 1.0 / (1.0 + dist_sum_3);
        const double q_numerator_value_4 = 1.0 / (1.0 + dist_sum_4);
        const double q_numerator_value_5 = 1.0 / (1.0 + dist_sum_5);
        const double q_numerator_value_6 = 1.0 / (1.0 + dist_sum_6);

        double q_value_1 = q_numerator_value_1;
        q_value_1 *= norm;
        if (q_value_1 < kMinimumProbability) {
          q_value_1 = kMinimumProbability;
        }
        double q_value_2 = q_numerator_value_2;
        q_value_2 *= norm;
        if (q_value_2 < kMinimumProbability) {
          q_value_2 = kMinimumProbability;
        }
        double q_value_3 = q_numerator_value_3;
        q_value_3 *= norm;
        if (q_value_3 < kMinimumProbability) {
          q_value_3 = kMinimumProbability;
        }
        double q_value_4 = q_numerator_value_4;
        q_value_4 *= norm;
        if (q_value_4 < kMinimumProbability) {
          q_value_4 = kMinimumProbability;
        }
        double q_value_5 = q_numerator_value_5;
        q_value_5 *= norm;
        if (q_value_5 < kMinimumProbability) {
          q_value_5 = kMinimumProbability;
        }
        double q_value_6 = q_numerator_value_6;
        q_value_6 *= norm;
        if (q_value_6 < kMinimumProbability) {
          q_value_6 = kMinimumProbability;
        }

        const double tmp_value_1 = (P[i * n + j] - q_value_1) * q_numerator_value_1;
        const double tmp_value_2 = (P[i * n + (j + 1)] - q_value_2) * q_numerator_value_2;
        const double tmp_value_3 = (P[i * n + (j + 2)] - q_value_3) * q_numerator_value_3;
        const double tmp_value_4 = (P[i * n + (j + 3)] - q_value_4) * q_numerator_value_4;
        const double tmp_value_5 = (P[i * n + (j + 4)] - q_value_5) * q_numerator_value_5;
        const double tmp_value_6 = (P[i * n + (j + 5)] - q_value_6) * q_numerator_value_6;
        const double value_1 = tmp_value_1 * (Y[i * m + l] - Y[j * m + l]);
        const double value_2 = tmp_value_2 * (Y[i * m + l] - Y[(j + 1) * m + l]);
        const double value_3 = tmp_value_3 * (Y[i * m + l] - Y[(j + 2) * m + l]);
        const double value_4 = tmp_value_4 * (Y[i * m + l] - Y[(j + 3) * m + l]);
        const double value_5 = tmp_value_5 * (Y[i * m + l] - Y[(j + 4) * m + l]);
        const double value_6 = tmp_value_6 * (Y[i * m + l] - Y[(j + 5) * m + l]);
        sum += value_1;
        sum += value_2;
        sum += value_3;
        sum += value_4;
        sum += value_5;
        sum += value_6;
      }
      // end
      for (int j = end; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double value = 4.0 * sum;
      grad_Y[i * m + l] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_unroll6(Matrix *Y, tsne_var_t *var, int n, int m,
                               double momentum) {
  grad_desc_no_vars_unroll6_pure(Y->data, var->P.data, var->grad_Y.data,
                                 var->Y_delta.data, var->gains.data, n, m,
                                 momentum);
}

void grad_desc_no_vars_unroll8_pure(double *Y, const double *P, double *grad_Y,
                                    double *Y_delta, double *gains, int n,
                                    int m, double momentum) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    constexpr int unroll_factor = 4;
    const int begin = (i + 4) / 4 * 4;  // first 32-byte aligned address after i
    const int end = begin + (n - begin) / unroll_factor * unroll_factor;
    // front
    for (int j = i + 1; j < begin; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
    // main
    for (int j = begin; j < end; j += unroll_factor) {
      double dist_sum_1 = 0.0;
      double dist_sum_2 = 0.0;
      double dist_sum_3 = 0.0;
      double dist_sum_4 = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist_1 = Y[i * m + k] - Y[j * m + k];
        const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
        const double dist_3 = Y[i * m + k] - Y[(j + 2) * m + k];
        const double dist_4 = Y[i * m + k] - Y[(j + 3) * m + k];
        dist_sum_1 += dist_1 * dist_1;
        dist_sum_2 += dist_2 * dist_2;
        dist_sum_3 += dist_3 * dist_3;
        dist_sum_4 += dist_4 * dist_4;
      }
      const double value_1 = 1.0 / (1.0 + dist_sum_1);
      const double value_2 = 1.0 / (1.0 + dist_sum_2);
      const double value_3 = 1.0 / (1.0 + dist_sum_3);
      const double value_4 = 1.0 / (1.0 + dist_sum_4);
      sum += value_1;
      sum += value_2;
      sum += value_3;
      sum += value_4;
    }
    // back
    for (int j = end; j < n; j++) {
      double dist_sum = 0.0;
      for (int k = 0; k < m; k++) {
        const double dist = Y[i * m + k] - Y[j * m + k];
        dist_sum += dist * dist;
      }
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    for (int l = 0; l < m; l++) {
      constexpr int unroll_factor = 8;
      const int end = n / unroll_factor * unroll_factor;
      sum = 0.0;
      // main
      for (int j = 0; j < end; j += unroll_factor) {
        double dist_sum_1 = 0.0;
        double dist_sum_2 = 0.0;
        double dist_sum_3 = 0.0;
        double dist_sum_4 = 0.0;
        double dist_sum_5 = 0.0;
        double dist_sum_6 = 0.0;
        double dist_sum_7 = 0.0;
        double dist_sum_8 = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist_1 = Y[i * m + k] - Y[j * m + k];
          const double dist_2 = Y[i * m + k] - Y[(j + 1) * m + k];
          const double dist_3 = Y[i * m + k] - Y[(j + 2) * m + k];
          const double dist_4 = Y[i * m + k] - Y[(j + 3) * m + k];
          const double dist_5 = Y[i * m + k] - Y[(j + 4) * m + k];
          const double dist_6 = Y[i * m + k] - Y[(j + 5) * m + k];
          const double dist_7 = Y[i * m + k] - Y[(j + 6) * m + k];
          const double dist_8 = Y[i * m + k] - Y[(j + 7) * m + k];
          dist_sum_1 += dist_1 * dist_1;
          dist_sum_2 += dist_2 * dist_2;
          dist_sum_3 += dist_3 * dist_3;
          dist_sum_4 += dist_4 * dist_4;
          dist_sum_5 += dist_5 * dist_5;
          dist_sum_6 += dist_6 * dist_6;
          dist_sum_7 += dist_7 * dist_7;
          dist_sum_8 += dist_8 * dist_8;
        }
        const double q_numerator_value_1 = 1.0 / (1.0 + dist_sum_1);
        const double q_numerator_value_2 = 1.0 / (1.0 + dist_sum_2);
        const double q_numerator_value_3 = 1.0 / (1.0 + dist_sum_3);
        const double q_numerator_value_4 = 1.0 / (1.0 + dist_sum_4);
        const double q_numerator_value_5 = 1.0 / (1.0 + dist_sum_5);
        const double q_numerator_value_6 = 1.0 / (1.0 + dist_sum_6);
        const double q_numerator_value_7 = 1.0 / (1.0 + dist_sum_7);
        const double q_numerator_value_8 = 1.0 / (1.0 + dist_sum_8);

        double q_value_1 = q_numerator_value_1;
        q_value_1 *= norm;
        if (q_value_1 < kMinimumProbability) {
          q_value_1 = kMinimumProbability;
        }
        double q_value_2 = q_numerator_value_2;
        q_value_2 *= norm;
        if (q_value_2 < kMinimumProbability) {
          q_value_2 = kMinimumProbability;
        }
        double q_value_3 = q_numerator_value_3;
        q_value_3 *= norm;
        if (q_value_3 < kMinimumProbability) {
          q_value_3 = kMinimumProbability;
        }
        double q_value_4 = q_numerator_value_4;
        q_value_4 *= norm;
        if (q_value_4 < kMinimumProbability) {
          q_value_4 = kMinimumProbability;
        }
        double q_value_5 = q_numerator_value_5;
        q_value_5 *= norm;
        if (q_value_5 < kMinimumProbability) {
          q_value_5 = kMinimumProbability;
        }
        double q_value_6 = q_numerator_value_6;
        q_value_6 *= norm;
        if (q_value_6 < kMinimumProbability) {
          q_value_6 = kMinimumProbability;
        }
        double q_value_7 = q_numerator_value_7;
        q_value_7 *= norm;
        if (q_value_7 < kMinimumProbability) {
          q_value_7 = kMinimumProbability;
        }
        double q_value_8 = q_numerator_value_8;
        q_value_8 *= norm;
        if (q_value_8 < kMinimumProbability) {
          q_value_8 = kMinimumProbability;
        }

        const double tmp_value_1 = (P[i * n + j] - q_value_1) * q_numerator_value_1;
        const double tmp_value_2 = (P[i * n + (j + 1)] - q_value_2) * q_numerator_value_2;
        const double tmp_value_3 = (P[i * n + (j + 2)] - q_value_3) * q_numerator_value_3;
        const double tmp_value_4 = (P[i * n + (j + 3)] - q_value_4) * q_numerator_value_4;
        const double tmp_value_5 = (P[i * n + (j + 4)] - q_value_5) * q_numerator_value_5;
        const double tmp_value_6 = (P[i * n + (j + 5)] - q_value_6) * q_numerator_value_6;
        const double tmp_value_7 = (P[i * n + (j + 6)] - q_value_7) * q_numerator_value_7;
        const double tmp_value_8 = (P[i * n + (j + 7)] - q_value_8) * q_numerator_value_8;
        const double value_1 = tmp_value_1 * (Y[i * m + l] - Y[j * m + l]);
        const double value_2 = tmp_value_2 * (Y[i * m + l] - Y[(j + 1) * m + l]);
        const double value_3 = tmp_value_3 * (Y[i * m + l] - Y[(j + 2) * m + l]);
        const double value_4 = tmp_value_4 * (Y[i * m + l] - Y[(j + 3) * m + l]);
        const double value_5 = tmp_value_5 * (Y[i * m + l] - Y[(j + 4) * m + l]);
        const double value_6 = tmp_value_6 * (Y[i * m + l] - Y[(j + 5) * m + l]);
        const double value_7 = tmp_value_7 * (Y[i * m + l] - Y[(j + 6) * m + l]);
        const double value_8 = tmp_value_8 * (Y[i * m + l] - Y[(j + 7) * m + l]);
        sum += value_1;
        sum += value_2;
        sum += value_3;
        sum += value_4;
        sum += value_5;
        sum += value_6;
        sum += value_7;
        sum += value_8;
      }
      // end
      for (int j = end; j < n; j++) {
        double dist_sum = 0.0;
        for (int k = 0; k < m; k++) {
          const double dist = Y[i * m + k] - Y[j * m + k];
          dist_sum += dist * dist;
        }
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y[i * m + l] - Y[j * m + l]);
        sum += value;
      }
      const double value = 4.0 * sum;
      grad_Y[i * m + l] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_unroll8(Matrix *Y, tsne_var_t *var, int n, int m,
                               double momentum) {
  grad_desc_no_vars_unroll8_pure(Y->data, var->P.data, var->grad_Y.data,
                                 var->Y_delta.data, var->gains.data, n, m,
                                 momentum);
}

void grad_desc_no_vars_fetch_pure(double *Y, const double *P, double *grad_Y,
                                  double *Y_delta, double *gains, int n, int m,
                                  double momentum) {
  assert(m == 2);

  double sum = 0;
  for (int i = 0; i < n; i++) {
    const double Y_i1 = Y[i * m];
    const double Y_i2 = Y[i * m + 1];
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      const double dist_1 = Y_i1 - Y[j * m];
      const double dist_2 = Y_i2 - Y[j * m + 1];
      dist_sum += dist_1 * dist_1;
      dist_sum += dist_2 * dist_2;
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    const double Y_i1 = Y[i * m];
    const double Y_i2 = Y[i * m + 1];
    for (int l = 0; l < m; l++) {
      const double Y_i_l = Y[i * m + l];
      sum = 0.0;
      for (int j = 0; j < n; j++) {
        double dist_sum = 0.0;
        const double dist_1 = Y_i1 - Y[j * m];
        const double dist_2 = Y_i2 - Y[j * m + 1];
        dist_sum += dist_1 * dist_1;
        dist_sum += dist_2 * dist_2;
        const double q_numerator_value = 1.0 / (1.0 + dist_sum);

        double q_value = q_numerator_value;
        q_value *= norm;
        if (q_value < kMinimumProbability) {
          q_value = kMinimumProbability;
        }

        const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
        const double value = tmp_value * (Y_i_l - Y[j * m + l]);
        sum += value;
      }
      const double value = 4.0 * sum;
      grad_Y[i * m + l] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_fetch(Matrix *Y, tsne_var_t *var, int n, int m,
                             double momentum) {
  grad_desc_no_vars_fetch_pure(Y->data, var->P.data, var->grad_Y.data,
                               var->Y_delta.data, var->gains.data, n, m,
                               momentum);
}

void grad_desc_no_vars_no_l_pure(double *Y, const double *P, double *grad_Y,
                                 double *Y_delta, double *gains, int n, int m,
                                 double momentum) {
  assert(m == 2);

  double sum = 0;
  for (int i = 0; i < n; i++) {
    const double Y_i1 = Y[i * m];
    const double Y_i2 = Y[i * m + 1];
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      const double dist_1 = Y_i1 - Y[j * m];
      const double dist_2 = Y_i2 - Y[j * m + 1];
      dist_sum += dist_1 * dist_1;
      dist_sum += dist_2 * dist_2;
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  for (int i = 0; i < n; i++) {
    const double Y_i1 = Y[i * m];
    const double Y_i2 = Y[i * m + 1];
    double sum_l1 = 0.0;
    double sum_l2 = 0.0;
    for (int j = 0; j < n; j++) {
      double dist_sum = 0.0;
      const double dist_k1 = Y_i1 - Y[j * m];
      const double dist_k2 = Y_i2 - Y[j * m + 1];
      dist_sum += dist_k1 * dist_k1;
      dist_sum += dist_k2 * dist_k2;
      const double q_numerator_value = 1.0 / (1.0 + dist_sum);

      double q_value = q_numerator_value;
      q_value *= norm;
      if (q_value < kMinimumProbability) {
        q_value = kMinimumProbability;
      }

      const double tmp_value = (P[i * n + j] - q_value) * q_numerator_value;
      const double value_l1 = tmp_value * dist_k1;
      const double value_l2 = tmp_value * dist_k2;
      sum_l1 += value_l1;
      sum_l2 += value_l2;
    }
    grad_Y[i * m ] = 4.0 * sum_l1;
    grad_Y[i * m + 1] = 4.0 * sum_l2;
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_no_l(Matrix *Y, tsne_var_t *var, int n, int m,
                            double momentum) {
  grad_desc_no_vars_no_l_pure(Y->data, var->P.data, var->grad_Y.data,
                              var->Y_delta.data, var->gains.data, n, m,
                              momentum);
}

void grad_desc_no_vars_unroll_pure(double *Y, const double *P, double *grad_Y,
                                 double *Y_delta, double *gains, int n, int m,
                                 double momentum) {
  assert(m == 2);

  double sum = 0;
  for (int i = 0; i < n; i++) {
    const double Y_i1 = Y[i * m];
    const double Y_i2 = Y[i * m + 1];
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      const double dist_1 = Y_i1 - Y[j * m];
      const double dist_2 = Y_i2 - Y[j * m + 1];
      dist_sum += dist_1 * dist_1;
      dist_sum += dist_2 * dist_2;
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const double norm = 0.5 / sum;

  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 2) {
    const double Y_i1[2] = {Y[i * 2], Y[i * 2 + 2]};
    const double Y_i2[2] = {Y[i * 2 + 1], Y[i * 2 + 3]};
    double sum_l1[2] = {0.0, 0.0};
    double sum_l2[2] = {0.0, 0.0};
    for (int j = 0; j < n; j++) {
      const double Y_j1 = Y[j * 2];
      const double Y_j2 = Y[j * 2 + 1];
      const double dist_k1[2] = {Y_i1[0] - Y_j1, Y_i1[1] - Y_j1};
      const double dist_k2[2] = {Y_i2[0] - Y_j2, Y_i2[1] - Y_j2};
      double dist_sum[2] = {0.0, 0.0};
      dist_sum[0] += dist_k1[0] * dist_k1[0];
      dist_sum[0] += dist_k2[0] * dist_k2[0];
      dist_sum[1] += dist_k1[1] * dist_k1[1];
      dist_sum[1] += dist_k2[1] * dist_k2[1];
      dist_sum[0] += 1.0;
      dist_sum[1] += 1.0;
      const double q_numerator_value[2] = {1.0 / dist_sum[0], 1.0 / dist_sum[1]};

      double q_value[2] = {q_numerator_value[0], q_numerator_value[1]};
      q_value[0] *= norm;
      if (q_value[0] < kMinimumProbability) {
        q_value[0] = kMinimumProbability;
      }
      q_value[1] *= norm;
      if (q_value[1] < kMinimumProbability) {
        q_value[1] = kMinimumProbability;
      }

      const double tmp_value[2] = {(P[i * n + j] - q_value[0]) * q_numerator_value[0], (P[(i + 1) * n + j] - q_value[1]) * q_numerator_value[1]};
      const double value_l1[2] = {tmp_value[0] * dist_k1[0], tmp_value[1] * dist_k1[1]};
      const double value_l2[2] = {tmp_value[0] * dist_k2[0], tmp_value[1] * dist_k2[1]};
      sum_l1[0] += value_l1[0];
      sum_l1[1] += value_l1[1];
      sum_l2[0] += value_l2[0];
      sum_l2[1] += value_l2[1];
    }
    grad_Y[i * 2] = 4.0 * sum_l1[0];
    grad_Y[i * 2 + 1] = 4.0 * sum_l2[0];
    grad_Y[i * 2 + 2] = 4.0 * sum_l1[1];
    grad_Y[i * 2 + 3] = 4.0 * sum_l2[1];
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_unroll(Matrix *Y, tsne_var_t *var, int n, int m,
                            double momentum) {
  grad_desc_no_vars_unroll_pure(Y->data, var->P.data, var->grad_Y.data,
                              var->Y_delta.data, var->gains.data, n, m,
                              momentum);
}

void grad_desc_no_vars_vector_pure(double *Y, const double *P, double *grad_Y,
                                   double *Y_delta, double *gains, int n, int m,
                                   double momentum) {
  assert(m == 2);

  double sum = 0;
  for (int i = 0; i < n; i++) {
    const double Y_i1 = Y[i * m];
    const double Y_i2 = Y[i * m + 1];
    for (int j = i + 1; j < n; j++) {
      double dist_sum = 0.0;
      const double dist_1 = Y_i1 - Y[j * m];
      const double dist_2 = Y_i2 - Y[j * m + 1];
      dist_sum += dist_1 * dist_1;
      dist_sum += dist_2 * dist_2;
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
  }

  const __m256d norm = _mm256_set1_pd(0.5 / sum);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d four = _mm256_set1_pd(4.0);
  const __m256d minimum_probability = _mm256_set1_pd(kMinimumProbability);

  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    const __m256d Y_i1 = _mm256_set_pd(Y[i * 2 + 6], Y[i * 2 + 4], Y[i * 2 + 2], Y[i * 2]);
    const __m256d Y_i2 = _mm256_set_pd(Y[i * 2 + 7], Y[i * 2 + 5], Y[i * 2 + 3], Y[i * 2 + 1]);
    __m256d sum_l1 = _mm256_setzero_pd();
    __m256d sum_l2 = _mm256_setzero_pd();
    for (int j = 0; j < n; j++) {
      const __m256d Y_j1 = _mm256_set1_pd(Y[j * 2]);
      const __m256d Y_j2 = _mm256_set1_pd(Y[j * 2 + 1]);
      const __m256d dist_k1 = Y_i1 - Y_j1;
      const __m256d dist_k2 = Y_i2 - Y_j2;
      __m256d dist_sum = _mm256_setzero_pd();
      dist_sum = _mm256_fmadd_pd(dist_k1, dist_k1, dist_sum);
      dist_sum = _mm256_fmadd_pd(dist_k2, dist_k2, dist_sum);
      dist_sum = _mm256_add_pd(dist_sum, one);
      const __m256d q_numerator_value = _mm256_div_pd(one, dist_sum);

      __m256d q_value = _mm256_mul_pd(q_numerator_value, norm);
      q_value = _mm256_max_pd(q_value, minimum_probability);

      const __m256d p = _mm256_set_pd(P[(i + 3) * n + j], P[(i + 2) * n + j], P[(i + 1) * n + j], P[i * n + j]);
      const __m256d sub = _mm256_sub_pd(p, q_value);

      const __m256d tmp_value = _mm256_mul_pd(sub, q_numerator_value);
      const __m256d value_l1 = tmp_value * dist_k1;
      const __m256d value_l2 = tmp_value * dist_k2;
      sum_l1 = _mm256_add_pd(sum_l1, value_l1);
      sum_l2 = _mm256_add_pd(sum_l2, value_l2);
    }
    sum_l1 = _mm256_mul_pd(sum_l1, four);
    sum_l2 = _mm256_mul_pd(sum_l2, four);
    double out[8];
    _mm256_store_pd(out, sum_l1);
    _mm256_store_pd(out + 4, sum_l2);
    grad_Y[i * 2] = out[0];
    grad_Y[i * 2 + 2] = out[1];
    grad_Y[i * 2 + 4] = out[2];
    grad_Y[i * 2 + 6] = out[3];
    grad_Y[i * 2 + 1] = out[4];
    grad_Y[i * 2 + 3] = out[5];
    grad_Y[i * 2 + 5] = out[6];
    grad_Y[i * 2 + 7] = out[7];
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_vector(Matrix *Y, tsne_var_t *var, int n, int m,
                            double momentum) {
  grad_desc_no_vars_vector_pure(Y->data, var->P.data, var->grad_Y.data,
                              var->Y_delta.data, var->gains.data, n, m,
                              momentum);
}

void grad_desc_no_vars_vector_acc_pure(double *Y, const double *P, double *grad_Y,
                                   double *Y_delta, double *gains, int n, int m,
                                   double momentum) {
  assert(m == 2);
  assert(n % 4 == 0);

  const __m256d one = _mm256_set1_pd(1.0);

  __m256d sum_acc = _mm256_setzero_pd();
  for (int i = 0; i < n; i += 4) {
    const __m256d Y_i1 = _mm256_set_pd(Y[i * 2 + 6], Y[i * 2 + 4], Y[i * 2 + 2], Y[i * 2]);
    const __m256d Y_i2 = _mm256_set_pd(Y[i * 2 + 7], Y[i * 2 + 5], Y[i * 2 + 3], Y[i * 2 + 1]);
    for (int j = i + 1; j < n; j++) {
      const __m256d Y_j1 = _mm256_set1_pd(Y[j * 2]);
      const __m256d Y_j2 = _mm256_set1_pd(Y[j * 2 + 1]);
      const __m256d dist_k1 = Y_i1 - Y_j1;
      const __m256d dist_k2 = Y_i2 - Y_j2;
      __m256d dist_sum = _mm256_setzero_pd();
      dist_sum = _mm256_fmadd_pd(dist_k1, dist_k1, dist_sum);
      dist_sum = _mm256_fmadd_pd(dist_k2, dist_k2, dist_sum);
      dist_sum = _mm256_add_pd(dist_sum, one);
      const __m256d q_numerator_value = _mm256_div_pd(one, dist_sum);
      sum_acc = _mm256_add_pd(sum_acc, q_numerator_value);
    }
  }
  double out[4];
  _mm256_store_pd(out, sum_acc);
  const double sum_scalar = out[0] + out[1] + out[2] + out[3];

  const __m256d norm = _mm256_set1_pd(0.5 / sum_scalar);
  const __m256d four = _mm256_set1_pd(4.0);
  const __m256d minimum_probability = _mm256_set1_pd(kMinimumProbability);

  for (int i = 0; i < n; i += 4) {
    const __m256d Y_i1 = _mm256_set_pd(Y[i * 2 + 6], Y[i * 2 + 4], Y[i * 2 + 2], Y[i * 2]);
    const __m256d Y_i2 = _mm256_set_pd(Y[i * 2 + 7], Y[i * 2 + 5], Y[i * 2 + 3], Y[i * 2 + 1]);
    __m256d sum_l1 = _mm256_setzero_pd();
    __m256d sum_l2 = _mm256_setzero_pd();
    for (int j = 0; j < n; j++) {
      const __m256d Y_j1 = _mm256_set1_pd(Y[j * 2]);
      const __m256d Y_j2 = _mm256_set1_pd(Y[j * 2 + 1]);
      const __m256d dist_k1 = Y_i1 - Y_j1;
      const __m256d dist_k2 = Y_i2 - Y_j2;
      __m256d dist_sum = _mm256_setzero_pd();
      dist_sum = _mm256_fmadd_pd(dist_k1, dist_k1, dist_sum);
      dist_sum = _mm256_fmadd_pd(dist_k2, dist_k2, dist_sum);
      dist_sum = _mm256_add_pd(dist_sum, one);
      const __m256d q_numerator_value = _mm256_div_pd(one, dist_sum);

      __m256d q_value = _mm256_mul_pd(q_numerator_value, norm);
      q_value = _mm256_max_pd(q_value, minimum_probability);

      const __m256d p = _mm256_set_pd(P[(i + 3) * n + j], P[(i + 2) * n + j], P[(i + 1) * n + j], P[i * n + j]);
      const __m256d sub = _mm256_sub_pd(p, q_value);

      const __m256d tmp_value = _mm256_mul_pd(sub, q_numerator_value);
      const __m256d value_l1 = tmp_value * dist_k1;
      const __m256d value_l2 = tmp_value * dist_k2;
      sum_l1 = _mm256_add_pd(sum_l1, value_l1);
      sum_l2 = _mm256_add_pd(sum_l2, value_l2);
    }
    sum_l1 = _mm256_mul_pd(sum_l1, four);
    sum_l2 = _mm256_mul_pd(sum_l2, four);
    double ooout[8];
    _mm256_store_pd(ooout, sum_l1);
    _mm256_store_pd(ooout + 4, sum_l2);
    grad_Y[i * 2] = ooout[0];
    grad_Y[i * 2 + 2] = ooout[1];
    grad_Y[i * 2 + 4] = ooout[2];
    grad_Y[i * 2 + 6] = ooout[3];
    grad_Y[i * 2 + 1] = ooout[4];
    grad_Y[i * 2 + 3] = ooout[5];
    grad_Y[i * 2 + 5] = ooout[6];
    grad_Y[i * 2 + 7] = ooout[7];
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_vector_acc(Matrix *Y, tsne_var_t *var, int n, int m,
                            double momentum) {
  grad_desc_no_vars_vector_acc_pure(Y->data, var->P.data, var->grad_Y.data,
                              var->Y_delta.data, var->gains.data, n, m,
                              momentum);
}

void grad_desc_no_vars_vector_inner_pure(double *Y, const double *P, double *grad_Y,
                                   double *Y_delta, double *gains, int n, int m,
                                   double momentum) {
  assert(m == 2);
  assert(n % 4 == 0);

  const __m256d one = _mm256_set1_pd(1.0);

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    const int begin = i / 4 * 4 + 4;
    for (int j = i + 1; j < begin; j++) {
      double dist_sum = 0.0;
      const double dist_1 = Y[i * m] - Y[j * m];
      const double dist_2 = Y[i * m + 1] - Y[j * m + 1];
      dist_sum += dist_1 * dist_1;
      dist_sum += dist_2 * dist_2;
      const double value = 1.0 / (1.0 + dist_sum);
      sum += value;
    }
    const __m256d Y_i1 = _mm256_set1_pd(Y[i * m]);
    const __m256d Y_i2 = _mm256_set1_pd(Y[i * m + 1]);
    for (int j = begin; j < n; j += 4) {
      const __m256d Y_j1 = _mm256_set_pd(Y[j * 2 + 6], Y[j * 2 + 4], Y[j * 2 + 2], Y[j * 2]);
      const __m256d Y_j2 = _mm256_set_pd(Y[j * 2 + 7], Y[j * 2 + 5], Y[j * 2 + 3], Y[j * 2 + 1]);
      const __m256d dist_k1 = Y_i1 - Y_j1;
      const __m256d dist_k2 = Y_i2 - Y_j2;
      __m256d dist_sum = _mm256_setzero_pd();
      dist_sum = _mm256_fmadd_pd(dist_k1, dist_k1, dist_sum);
      dist_sum = _mm256_fmadd_pd(dist_k2, dist_k2, dist_sum);
      dist_sum = _mm256_add_pd(dist_sum, one);
      const __m256d value = _mm256_div_pd(one, dist_sum);
      double out[4];
      _mm256_store_pd(out, value);
      sum += out[0];
      sum += out[1];
      sum += out[2];
      sum += out[3];
    }
  }

  const __m256d norm = _mm256_set1_pd(0.5 / sum);
  const __m256d four = _mm256_set1_pd(4.0);
  const __m256d minimum_probability = _mm256_set1_pd(kMinimumProbability);

  for (int i = 0; i < n; i += 4) {
    const __m256d Y_i1 = _mm256_set_pd(Y[i * 2 + 6], Y[i * 2 + 4], Y[i * 2 + 2], Y[i * 2]);
    const __m256d Y_i2 = _mm256_set_pd(Y[i * 2 + 7], Y[i * 2 + 5], Y[i * 2 + 3], Y[i * 2 + 1]);
    __m256d sum_l1 = _mm256_setzero_pd();
    __m256d sum_l2 = _mm256_setzero_pd();
    for (int j = 0; j < n; j++) {
      const __m256d Y_j1 = _mm256_set1_pd(Y[j * 2]);
      const __m256d Y_j2 = _mm256_set1_pd(Y[j * 2 + 1]);
      const __m256d dist_k1 = Y_i1 - Y_j1;
      const __m256d dist_k2 = Y_i2 - Y_j2;
      __m256d dist_sum = _mm256_setzero_pd();
      dist_sum = _mm256_fmadd_pd(dist_k1, dist_k1, dist_sum);
      dist_sum = _mm256_fmadd_pd(dist_k2, dist_k2, dist_sum);
      dist_sum = _mm256_add_pd(dist_sum, one);
      const __m256d q_numerator_value = _mm256_div_pd(one, dist_sum);

      __m256d q_value = _mm256_mul_pd(q_numerator_value, norm);
      q_value = _mm256_max_pd(q_value, minimum_probability);

      const __m256d p = _mm256_set_pd(P[(i + 3) * n + j], P[(i + 2) * n + j], P[(i + 1) * n + j], P[i * n + j]);
      const __m256d sub = _mm256_sub_pd(p, q_value);

      const __m256d tmp_value = _mm256_mul_pd(sub, q_numerator_value);
      const __m256d value_l1 = tmp_value * dist_k1;
      const __m256d value_l2 = tmp_value * dist_k2;
      sum_l1 = _mm256_add_pd(sum_l1, value_l1);
      sum_l2 = _mm256_add_pd(sum_l2, value_l2);
    }
    sum_l1 = _mm256_mul_pd(sum_l1, four);
    sum_l2 = _mm256_mul_pd(sum_l2, four);
    double out[8];
    _mm256_store_pd(out, sum_l1);
    _mm256_store_pd(out + 4, sum_l2);
    grad_Y[i * 2] = out[0];
    grad_Y[i * 2 + 2] = out[1];
    grad_Y[i * 2 + 4] = out[2];
    grad_Y[i * 2 + 6] = out[3];
    grad_Y[i * 2 + 1] = out[4];
    grad_Y[i * 2 + 3] = out[5];
    grad_Y[i * 2 + 5] = out[6];
    grad_Y[i * 2 + 7] = out[7];
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_vector_inner(Matrix *Y, tsne_var_t *var, int n, int m,
                            double momentum) {
  grad_desc_no_vars_vector_inner_pure(Y->data, var->P.data, var->grad_Y.data,
                              var->Y_delta.data, var->gains.data, n, m,
                              momentum);
}

void grad_desc_no_vars_vector_unroll2_pure(double *Y, const double *P, double *grad_Y,
                                   double *Y_delta, double *gains, int n, int m,
                                   double momentum) {
  assert(m == 2);
  assert(n % 8 == 0);

  const __m256d one = _mm256_set1_pd(1.0);

  __m256d sum_acc = _mm256_setzero_pd();
  for (int i = 0; i < n; i += 4) {
    const __m256d Y_i1 = _mm256_set_pd(Y[i * 2 + 6], Y[i * 2 + 4], Y[i * 2 + 2], Y[i * 2]);
    const __m256d Y_i2 = _mm256_set_pd(Y[i * 2 + 7], Y[i * 2 + 5], Y[i * 2 + 3], Y[i * 2 + 1]);
    for (int j = i + 1; j < n; j++) {
      const __m256d Y_j1 = _mm256_set1_pd(Y[j * 2]);
      const __m256d Y_j2 = _mm256_set1_pd(Y[j * 2 + 1]);
      const __m256d dist_k1 = Y_i1 - Y_j1;
      const __m256d dist_k2 = Y_i2 - Y_j2;
      __m256d dist_sum = _mm256_setzero_pd();
      dist_sum = _mm256_fmadd_pd(dist_k1, dist_k1, dist_sum);
      dist_sum = _mm256_fmadd_pd(dist_k2, dist_k2, dist_sum);
      dist_sum = _mm256_add_pd(dist_sum, one);
      const __m256d q_numerator_value = _mm256_div_pd(one, dist_sum);
      sum_acc = _mm256_add_pd(sum_acc, q_numerator_value);
    }
  }
  double out[4];
  _mm256_store_pd(out, sum_acc);
  const double sum_scalar = out[0] + out[1] + out[2] + out[3];

  const __m256d norm = _mm256_set1_pd(0.5 / sum_scalar);
  const __m256d four = _mm256_set1_pd(4.0);
  const __m256d minimum_probability = _mm256_set1_pd(kMinimumProbability);

  for (int i = 0; i < n; i += 8) {
    const __m256d Y_i1_1 = _mm256_set_pd(Y[i * 2 + 6], Y[i * 2 + 4], Y[i * 2 + 2], Y[i * 2]);
    const __m256d Y_i2_1 = _mm256_set_pd(Y[i * 2 + 7], Y[i * 2 + 5], Y[i * 2 + 3], Y[i * 2 + 1]);
    const __m256d Y_i1_2 = _mm256_set_pd(Y[i * 2 + 14], Y[i * 2 + 12], Y[i * 2 + 10], Y[i * 2 + 8]);
    const __m256d Y_i2_2 = _mm256_set_pd(Y[i * 2 + 15], Y[i * 2 + 13], Y[i * 2 + 11], Y[i * 2 + 9]);
    __m256d sum_l1_1 = _mm256_setzero_pd();
    __m256d sum_l2_1 = _mm256_setzero_pd();
    __m256d sum_l1_2 = _mm256_setzero_pd();
    __m256d sum_l2_2 = _mm256_setzero_pd();
    for (int j = 0; j < n; j++) {
      const __m256d Y_j1 = _mm256_set1_pd(Y[j * 2]);
      const __m256d Y_j2 = _mm256_set1_pd(Y[j * 2 + 1]);
      const __m256d dist_k1_1 = Y_i1_1 - Y_j1;
      const __m256d dist_k2_1 = Y_i2_1 - Y_j2;
      const __m256d dist_k1_2 = Y_i1_2 - Y_j1;
      const __m256d dist_k2_2 = Y_i2_2 - Y_j2;
      __m256d dist_sum_1 = _mm256_setzero_pd();
      dist_sum_1 = _mm256_fmadd_pd(dist_k1_1, dist_k1_1, dist_sum_1);
      dist_sum_1 = _mm256_fmadd_pd(dist_k2_1, dist_k2_1, dist_sum_1);
      dist_sum_1 = _mm256_add_pd(dist_sum_1, one);
      __m256d dist_sum_2 = _mm256_setzero_pd();
      dist_sum_2 = _mm256_fmadd_pd(dist_k1_2, dist_k1_2, dist_sum_2);
      dist_sum_2 = _mm256_fmadd_pd(dist_k2_2, dist_k2_2, dist_sum_2);
      dist_sum_2 = _mm256_add_pd(dist_sum_2, one);
      const __m256d q_numerator_value_1 = _mm256_div_pd(one, dist_sum_1);
      const __m256d q_numerator_value_2 = _mm256_div_pd(one, dist_sum_2);

      __m256d q_value_1 = _mm256_mul_pd(q_numerator_value_1, norm);
      __m256d q_value_2 = _mm256_mul_pd(q_numerator_value_2, norm);
      q_value_1 = _mm256_max_pd(q_value_1, minimum_probability);
      q_value_2 = _mm256_max_pd(q_value_2, minimum_probability);

      const __m256d p_1 = _mm256_set_pd(P[(i + 3) * n + j], P[(i + 2) * n + j], P[(i + 1) * n + j], P[i * n + j]);
      const __m256d p_2 = _mm256_set_pd(P[(i + 7) * n + j], P[(i + 6) * n + j], P[(i + 5) * n + j], P[(i + 4) * n + j]);
      const __m256d sub_1 = _mm256_sub_pd(p_1, q_value_1);
      const __m256d sub_2 = _mm256_sub_pd(p_2, q_value_2);

      const __m256d tmp_value_1 = _mm256_mul_pd(sub_1, q_numerator_value_1);
      const __m256d tmp_value_2 = _mm256_mul_pd(sub_2, q_numerator_value_2);
      const __m256d value_l1_1 = tmp_value_1 * dist_k1_1;
      const __m256d value_l2_1 = tmp_value_1 * dist_k2_1;
      const __m256d value_l1_2 = tmp_value_2 * dist_k1_2;
      const __m256d value_l2_2 = tmp_value_2 * dist_k2_2;
      sum_l1_1 = _mm256_add_pd(sum_l1_1, value_l1_1);
      sum_l2_1 = _mm256_add_pd(sum_l2_1, value_l2_1);
      sum_l1_2 = _mm256_add_pd(sum_l1_2, value_l1_2);
      sum_l2_2 = _mm256_add_pd(sum_l2_2, value_l2_2);
    }
    sum_l1_1 = _mm256_mul_pd(sum_l1_1, four);
    sum_l2_1 = _mm256_mul_pd(sum_l2_1, four);
    sum_l1_2 = _mm256_mul_pd(sum_l1_2, four);
    sum_l2_2 = _mm256_mul_pd(sum_l2_2, four);
    double ooout[16];
    _mm256_store_pd(ooout, sum_l1_1);
    _mm256_store_pd(ooout + 4, sum_l2_1);
    _mm256_store_pd(ooout + 8, sum_l1_2);
    _mm256_store_pd(ooout + 12, sum_l2_2);
    grad_Y[i * 2] = ooout[0];
    grad_Y[i * 2 + 2] = ooout[1];
    grad_Y[i * 2 + 4] = ooout[2];
    grad_Y[i * 2 + 6] = ooout[3];

    grad_Y[i * 2 + 1] = ooout[4];
    grad_Y[i * 2 + 3] = ooout[5];
    grad_Y[i * 2 + 5] = ooout[6];
    grad_Y[i * 2 + 7] = ooout[7];

    grad_Y[i * 2 + 8] = ooout[8];
    grad_Y[i * 2 + 10] = ooout[9];
    grad_Y[i * 2 + 12] = ooout[10];
    grad_Y[i * 2 + 14] = ooout[11];

    grad_Y[i * 2 + 9] = ooout[12];
    grad_Y[i * 2 + 11] = ooout[13];
    grad_Y[i * 2 + 13] = ooout[14];
    grad_Y[i * 2 + 15] = ooout[15];
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const bool positive_grad = (grad_Y[i * m + j] > 0);
      const bool positive_delta = (Y_delta[i * m + j] > 0);
      double value = gains[i * m + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) {
        value = kMinGain;
      }
      gains[i * m + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      const double value = momentum * Y_delta[i * m + j] -
                           kEta * gains[i * m + j] * grad_Y[i * m + j];
      Y_delta[i * m + j] = value;
      Y[i * m + j] += value;
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
      means[j] += Y[i * m + j];
    }
  }
  // take mean
  for (int j = 0; j < m; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      Y[i * m + j] -= means[j];
    }
  }
}

void grad_desc_no_vars_vector_unroll2(Matrix *Y, tsne_var_t *var, int n, int m,
                            double momentum) {
  grad_desc_no_vars_vector_unroll2_pure(Y->data, var->P.data, var->grad_Y.data,
                              var->Y_delta.data, var->gains.data, n, m,
                              momentum);
}

void tsne_no_vars(Matrix *X, Matrix *Y, tsne_var_t *var, int m) {
  int n = X->nrows;

  joint_probs_avx_fma_acc4(X, &var->P, &var->D);

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

    grad_desc_no_vars_vector_acc(Y, var, n, m, momentum);
  }
}
