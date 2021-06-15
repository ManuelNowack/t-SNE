#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tsne/func_registry.h"
#include "tsne/hyperparams.h"
#include "tsne/matrix.h"
#include "tsc_x86.h"

#define CYCLES_REQUIRED 1000000000
#define REP kGradDescMaxIter


double perf_grad_desc(grad_desc_func_t *f, const Matrix *Y, tsne_var_t *var) {
  const int n = Y->nrows;
  const int m = Y->ncols;

  Matrix Y_copy;
  copy_matrix(Y, &Y_copy);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      var->Y_delta.data[i * m + j] = 0.0;
      var->gains.data[i * m + j] = 1.0;
    }
  }

  uint64_t total_cycles = 0;

  double momentum = kInitialMomentum;
  for (int iter = 0; iter < REP; iter++) {
    if (iter == 100) {
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double value = var->P.data[i * n + j] / 4.0;
          var->P.data[i * n + j] = value;
          var->P.data[j * n + i] = value;
        }
      }
    }

    if (iter == 20) {
      momentum = kFinalMomentum;
    }

    uint64_t start = start_tsc();
    f(&Y_copy, var, n, m, momentum);
    uint64_t end = stop_tsc(start);
    total_cycles += end;
  }

  return double(total_cycles) / REP;
}


int main(int argc, char const *argv[])
{
  if (argc < 3) {
    printf("Usage: %s X_PCA Y_INIT\n", argv[0]);
    return 1;
  }

  Matrix X, Y;
  load_matrix(argv[1], &X);
  load_matrix(argv[2], &Y);

  const int n = Y.nrows;
  const int m = Y.ncols;
  tsne_var_t var;
  create_tsne_variables(var, n, m);

  joint_probs_baseline(&X, &var.P, &var.D);

  double cycles;
  cycles = perf_grad_desc(grad_desc_baseline, &Y, &var);
  printf("grad_desc_baseline %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_baseline, &Y, &var);
  printf("grad_desc_no_vars_baseline %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_tmp, &Y, &var);
  printf("grad_desc_no_vars_tmp %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_D, &Y, &var);
  printf("grad_desc_no_vars_D %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_Q, &Y, &var);
  printf("grad_desc_no_vars_Q %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_Q_numerators, &Y, &var);
  printf("grad_desc_no_vars_Q_numerators %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_scalar, &Y, &var);
  printf("grad_desc_no_vars_scalar %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_no_if, &Y, &var);
  printf("grad_desc_no_vars_no_if %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_unroll2, &Y, &var);
  printf("grad_desc_no_vars_unroll2 %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_unroll4, &Y, &var);
  printf("grad_desc_no_vars_unroll4 %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_unroll6, &Y, &var);
  printf("grad_desc_no_vars_unroll6 %e\n", cycles);

  cycles = perf_grad_desc(grad_desc_no_vars_unroll8, &Y, &var);
  printf("grad_desc_no_vars_unroll8 %e\n", cycles);

  destroy_tsne_variables(var);
}