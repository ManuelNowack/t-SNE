#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tsne/matrix.h"
#include "tsc_x86.h"

/** IMPORTANT
 *  affinities.cpp must be compiled with the macro MY_EUCLIDEAN_DIST defined as
 *  a no-op or computing the euclidean distances will be included in the
 *  benchmark.
 **/

#define REP 1000

typedef void affinities_func_t(Matrix *Y, Matrix *Q, Matrix *Q_numerators,
                               Matrix *D);
typedef void euclidean_dist_func_t(Matrix *X, Matrix *D);

euclidean_dist_func_t euclidean_dist_baseline;

affinities_func_t affinities_baseline;
affinities_func_t affinities_no_triangle;
affinities_func_t affinities_no_Q_numerators;
affinities_func_t affinities_unroll_fst_4;
affinities_func_t affinities_unroll_snd_4;
affinities_func_t affinities_unroll_snd_8;
affinities_func_t affinities_unroll_both;
affinities_func_t affinities_vectorization;
affinities_func_t affinities_vectorization_no_Q_numerators;
affinities_func_t affinities_vectorization_4;
affinities_func_t affinities_accumulator;


double perf_affinities(affinities_func_t *f, Matrix *X, Matrix *Y) {
  int n = X->nrows;
  const int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  euclidean_dist_baseline(Y, &var.D);

  uint64_t total_cycles = 0;
  for (int i = 0; i < REP; ++i) {
    uint64_t start = start_tsc();
    f(Y, &var.Q, &var.Q_numerators, &var.D);
    uint64_t end = stop_tsc(start);
    total_cycles += end;
  }
  destroy_tsne_variables(var);

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

  double cycles;
  cycles = perf_affinities(affinities_baseline, &X, &Y);
  printf("baseline %e\n", cycles);

  cycles = perf_affinities(affinities_no_triangle, &X, &Y);
  printf("no triangle %e\n", cycles);

  cycles = perf_affinities(affinities_no_Q_numerators, &X, &Y);
  printf("no Q_numerators %e\n", cycles);

  cycles = perf_affinities(affinities_unroll_fst_4, &X, &Y);
  printf("unroll 4 fst %e\n", cycles);

  cycles = perf_affinities(affinities_unroll_snd_4, &X, &Y);
  printf("unroll 4 snd %e\n", cycles);

  cycles = perf_affinities(affinities_unroll_snd_8, &X, &Y);
  printf("unroll 8 snd %e\n", cycles);

  cycles = perf_affinities(affinities_unroll_both, &X, &Y);
  printf("unroll both %e\n", cycles);

  cycles = perf_affinities(affinities_vectorization, &X, &Y);
  printf("vectorization %e\n", cycles);

  cycles = perf_affinities(affinities_vectorization_no_Q_numerators, &X, &Y);
  printf("vectorization without Q_numerators %e\n", cycles);

  cycles = perf_affinities(affinities_vectorization_4, &X, &Y);
  printf("vectorization 4 %e\n", cycles);

  cycles = perf_affinities(affinities_accumulator, &X, &Y);
  printf("accumulator %e\n", cycles);
}