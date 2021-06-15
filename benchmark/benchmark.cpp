#include "benchmark.h"

#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

#include "tsc_x86.h"

#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"

// Computes and reports the number of cycles required per iteration
// for the given tsne function.
double perf_test_tsne(tsne_func_t *f, Matrix &X, Matrix &Y) {
  double cycles = 0.;
  size_t num_runs = 1;
  double multiplier = 1;
  uint64_t start, end;

  int n = X.nrows;
  const int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(&X, &Y, &var, n_dim);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);

  // Actual performance measurements repeated REP times.
  // We simply store all results and compute medians during post-processing.
  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(&X, &Y, &var, n_dim);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;
    total_cycles += cycles;
  }
  total_cycles /= REP;

  cycles = total_cycles;
  destroy_tsne_variables(var);

  return cycles;
}

// Computes and reports the number of cycles required per iteration
// for the given joint probabilities function.
double perf_test_joint_probs(joint_probs_func_t *f, Matrix &X) {
  double cycles = 0.;
  size_t num_runs = 1;
  double multiplier = 1;
  uint64_t start, end;

  int n = X.nrows;
  const int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(&X, &var.P, &var.D);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);

  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(&X, &var.P, &var.D);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;
    total_cycles += cycles;
  }
  total_cycles /= REP;

  cycles = total_cycles;
  destroy_tsne_variables(var);

  return cycles;
}

// Computes and reports the number of cycles required per iteration
// for the given joint probabilities function.
double perf_test_grad_desc(grad_desc_func_t *f, Matrix &X, Matrix &Y) {
  double cycles = 0.;
  size_t num_runs = 1;
  double multiplier = 1;
  uint64_t start, end;

  int n = X.nrows;
  const int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  // Populate the joint probability matrix.
  joint_probs_baseline(&X, &var.P, &var.D);

  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(&Y, &var, n, n_dim, kFinalMomentum);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);

  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(&Y, &var, n, n_dim, kFinalMomentum);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;
    total_cycles += cycles;
  }
  total_cycles /= REP;

  cycles = total_cycles;
  destroy_tsne_variables(var);

  return cycles;
}

double perf_test_log_perplexity(log_perplexity_func_t *f, Matrix &X) {
  double cycles = 0.;
  size_t num_runs = 1;
  double multiplier = 1;
  uint64_t start, end;

  int n = X.nrows;
  const int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  // Populate the distance matrix.
  euclidean_dist_baseline(&X, &var.D);

  double *distances = var.D.data;
  double *probs = var.P.data;
  double log_perp, norm;

  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(distances, probs, n, 0, 0.5, &log_perp, &norm);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);
  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(distances, probs, n, 0, 0.5, &log_perp, &norm);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;
    total_cycles += cycles;
  }
  total_cycles /= REP;

  cycles = total_cycles;
  destroy_tsne_variables(var);

  return cycles;
}

// Computes and reports the number of cycles required per iteration
// for the given squared Euclidean distance function.
double perf_test_euclidean_dist(euclidean_dist_func_t *f, Matrix &X) {
  double cycles = 0.;
  size_t num_runs = 1;
  double multiplier = 1;
  uint64_t start, end;

  int n = X.nrows;
  const int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(&X, &var.D);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);

  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(&X, &var.D);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;
    total_cycles += cycles;
  }
  total_cycles /= REP;

  cycles = total_cycles;
  destroy_tsne_variables(var);

  return cycles;
}
