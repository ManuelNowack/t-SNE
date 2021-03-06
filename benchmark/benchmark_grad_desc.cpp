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

int pow(int base, int exponent) {
  int power = 1;
  for (int i = 0; i < exponent; ++i)
  {
    power *= base;
  }
  return power;
}


int main(int argc, char const *argv[])
{
  if (argc < 3 || argc > 5) {
    printf("Usage: %s X_PCA Y_INIT [min samples] [max samples]\n", argv[0]);
    return 1;
  }

  Matrix X, Y;
  load_matrix(argv[1], &X);
  load_matrix(argv[2], &Y);

  const int n = Y.nrows;
  const int m = Y.ncols;
  int n_min, n_max;
  if (argc == 3) {
    n_min = n;
    n_max = n;
  }
  else if (argc == 4) {
    n_min = atoi(argv[3]);
    n_max = n_min;
  }
  else if (argc == 5) {
    n_min = pow(2, atoi(argv[3]));
    n_max = pow(2, atoi(argv[4]));
  }
  if (n_min < 1)
  {
    printf("%d samples are requested but at least one is required", n_min);
    return 1;
  }
  if (n_max > n)
  {
    printf("%d samples are requested but only %d are available", n_max, n);
    return 1;
  }

  for (int i = n_min; i <= n_max; i *= 2)
  {
    X.nrows = i;
    Y.nrows = i;
    tsne_var_t var;
    create_tsne_variables(var, i, m);

    joint_probs_baseline(&X, &var.P, &var.D);

    printf("%d samples\n\n", i);

    double cycles;
    cycles = perf_grad_desc(grad_desc_baseline, &Y, &var);
    printf("grad_desc_baseline %e\n", cycles);

    cycles = perf_grad_desc(_grad_desc_vec, &Y, &var);
    printf("_grad_desc_vec %e\n", cycles);

    cycles = perf_grad_desc(_grad_desc_vec2, &Y, &var);
    printf("_grad_desc_vec2 %e\n", cycles);

    cycles = perf_grad_desc(_grad_desc_vec3, &Y, &var);
    printf("_grad_desc_vec3 %e\n", cycles);

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

    cycles = perf_grad_desc(grad_desc_no_vars_grad, &Y, &var);
    printf("grad_desc_no_vars_grad %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_means, &Y, &var);
    printf("grad_desc_no_vars_means %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_unroll2, &Y, &var);
    printf("grad_desc_no_vars_unroll2 %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_unroll4, &Y, &var);
    printf("grad_desc_no_vars_unroll4 %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_unroll6, &Y, &var);
    printf("grad_desc_no_vars_unroll6 %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_unroll8, &Y, &var);
    printf("grad_desc_no_vars_unroll8 %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_fetch, &Y, &var);
    printf("grad_desc_no_vars_fetch %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_no_l, &Y, &var);
    printf("grad_desc_no_vars_no_l %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_unroll, &Y, &var);
    printf("grad_desc_no_vars_unroll %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_vector, &Y, &var);
    printf("grad_desc_no_vars_vector %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_vector_acc, &Y, &var);
    printf("grad_desc_no_vars_vector_acc %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_vector_inner, &Y, &var);
    printf("grad_desc_no_vars_vector_inner %e\n", cycles);

    cycles = perf_grad_desc(grad_desc_no_vars_vector_unroll2, &Y, &var);
    printf("grad_desc_no_vars_vector_unroll2 %e\n", cycles);

    destroy_tsne_variables(var);
  }
}