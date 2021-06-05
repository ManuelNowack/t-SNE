#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

void grad_desc_less_matrices(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                             double momentum) {
  // START: Euclidean Distances
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n_dim; k++) {
        const double dist = Y->data[i * n_dim + k] - Y->data[j * n_dim + k];
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

  // calculate gradient with respect to embeddings Y
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
                     var->Q_numerators.data[i * n + j];
      var->tmp.data[i * n + j] = value;
      var->tmp.data[j * n + i] = value;
    }
    var->tmp.data[i * n + i] = 0.0;
  }
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n_dim; k++) {
      double value = 0;
      for (int j = 0; j < n; j++) {
        value += var->tmp.data[i * n + j] *
                 (Y->data[i * n_dim + k] - Y->data[j * n_dim + k]);
      }
      value *= 4;
      var->grad_Y.data[i * n_dim + k] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      bool positive_grad = (var->grad_Y.data[i * n_dim + j] > 0);
      bool positive_delta = (var->Y_delta.data[i * n_dim + j] > 0);
      double value = var->gains.data[i * n_dim + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) value = kMinGain;
      var->gains.data[i * n_dim + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      double value = momentum * var->Y_delta.data[i * n_dim + j] -
                     kEta * var->gains.data[i * n_dim + j] *
                         var->grad_Y.data[i * n_dim + j];
      var->Y_delta.data[i * n_dim + j] = value;
      Y->data[i * n_dim + j] += value;
    }
  }

  // center each dimension at 0
  double means[n_dim];
  for (int j = 0; j < n_dim; j++) {
    means[j] = 0;
  }
  // accumulate
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      means[j] += Y->data[i * n_dim + j];
    }
  }
  // take mean
  for (int j = 0; j < n_dim; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      Y->data[i * n_dim + j] -= means[j];
    }
  }
}

void tsne_less_matrices_baseline(Matrix *X, Matrix *Y, tsne_var_t *var,
                                 int n_dim) {
  int n = X->nrows;

  joint_probs_baseline(X, &var->P, &var->D);

  // determine embeddings
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      var->Y_delta.data[i * n_dim + j] = 0.0;
      var->gains.data[i * n_dim + j] = 1.0;
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

    grad_desc_less_matrices(Y, var, n, n_dim, momentum);
  }
}
