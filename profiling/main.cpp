#include <tsne/matrix.h>

#include <stdio.h>
#include <stdlib.h>

#define REP 5

void tsne_baseline(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim);

void create_tsne_variables(tsne_var_t &var, int n, int n_dim) {
  var.P = create_matrix(n, n);
  var.Q = create_matrix(n, n);
  var.Q_numerators = create_matrix(n, n);
  var.grad_Y = create_matrix(n, n_dim);
  var.Y_delta = create_matrix(n, n_dim);
  var.tmp = create_matrix(n, n);
  var.gains = create_matrix(n, n_dim);
  var.D = create_matrix(n, n);
}

void destroy_tsne_variables(tsne_var_t &var) {
  free(var.P.data);
  free(var.Q.data);
  free(var.Q_numerators.data);
  free(var.grad_Y.data);
  free(var.Y_delta.data);
  free(var.tmp.data);
  free(var.gains.data);
  free(var.D.data);
}


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
    tsne_baseline(&X, &Y, &var, n_dim);

  }
  destroy_tsne_variables(var);

  return 0;
}
