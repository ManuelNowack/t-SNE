#include <tsne/grad_descent.h>
#include <tsne/baseline.h>

void grad_desc_baseline(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum) {
  // calculate low-dimensional affinities
  calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);

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
      double value =
          momentum * var->Y_delta.data[i * n_dim + j] -
          kEta * var->gains.data[i * n_dim + j] * var->grad_Y.data[i * n_dim + j];
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

void grad_desc_ndim_unroll(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum) {
  // calculate low-dimensional affinities
  calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);

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
      double value =
          momentum * var->Y_delta.data[i * n_dim + j] -
          kEta * var->gains.data[i * n_dim + j] * var->grad_Y.data[i * n_dim + j];
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