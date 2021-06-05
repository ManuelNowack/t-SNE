#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

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

    grad_desc_baseline(Y, var, n, n_dim, momentum);
  }
}
