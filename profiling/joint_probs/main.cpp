#include <tsne/matrix.h>
#include <tsne/hyperparams.h>
#include <tsne/func_registry.h>

#include <stdio.h>
#include <stdlib.h>

#define REP 3

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s X_PCA Y_INIT\n", argv[0]);
    return 1;
  }

  Matrix X = load_matrix(argv[1]);
  Matrix Y = load_matrix(argv[2]);

  int n = X.nrows;
  int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  for (int i = 0; i < REP; ++i) {
    joint_probs_avx_fma_acc4(&X, &var.P, &var.D);
  }
  destroy_tsne_variables(var);

  return 0;
}
