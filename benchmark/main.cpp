/**
 *      _________   _____________________  ____  ______
 *     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
 *    / /_  / /| | \__ \ / / / /   / / / / / / / __/
 *   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
 *  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
 *
 *  http://www.acl.inf.ethz.ch/teaching/fastcode
 *  How to Write Fast Numerical Code 263-2300 - ETH Zurich
 *  Copyright (C) 2019
 *                   Tyler Smith        (smitht@inf.ethz.ch)
 *                   Alen Stojanov      (astojanov@inf.ethz.ch)
 *                   Gagandeep Singh    (gsingh@inf.ethz.ch)
 *                   Markus Pueschel    (pueschel@inf.ethz.ch)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include <tsne/func_registry.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tsc_x86.h"

using namespace std;

#define CYCLES_REQUIRED (1 * 1e8)
#define REP 5  // 50

// Create intermediate t-SNE variables.
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

// Computes and reports the number of cycles required per iteration
// for the given tsne function.
double perf_test_tsne(tsne_func_t *f, const Matrix &X, Matrix &Y) {
  double cycles = 0.;
  long num_runs = 1;
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
      f(X, Y, var, n_dim);
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
      f(X, Y, var, n_dim);
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
double perf_test_joint_probs(joint_probs_func_t *f, const Matrix &X) {
  double cycles = 0.;
  long num_runs = 1;
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
      f(X, var.P, var.D);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);

  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(X, var.P, var.D);
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
double perf_test_grad_desc(grad_desc_func_t *f, joint_probs_func_t *joint_probs,
                           const Matrix &X, Matrix &Y) {
  double cycles = 0.;
  long num_runs = 1;
  double multiplier = 1;
  uint64_t start, end;

  int n = X.nrows;
  const int n_dim = 2;
  tsne_var_t var;
  create_tsne_variables(var, n, n_dim);

  // Populate the joint probability matrix.
  joint_probs(X, var.P, var.D);

  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(Y, var, n, n_dim, kFinalMomentum);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);

  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(Y, var, n, n_dim, kFinalMomentum);
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

int main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " X_PCA Y_INIT" << endl;
    return 1;
  }

  Matrix X = load_matrix(argv[1]);
  Matrix Y = load_matrix(argv[2]);

  register_functions();
  auto &tsne_func_registry = FuncResitry<tsne_func_t>::get_instance();
  auto &joint_probs_func_registry =
      FuncResitry<joint_probs_func_t>::get_instance();
  auto &grad_desc_func_registry = FuncResitry<grad_desc_func_t>::get_instance();

  // TODO(mrettenba): Check validity of functions.

  double perf;
  for (int i = 0; i < tsne_func_registry.num_funcs; i++) {
    perf = perf_test_tsne(tsne_func_registry.funcs[i], X, Y);
    cout << tsne_func_registry.func_names[i] << "," << perf << endl;
  }

  for (int i = 0; i < joint_probs_func_registry.num_funcs; i++) {
    perf = perf_test_joint_probs(joint_probs_func_registry.funcs[i], X);
    cout << joint_probs_func_registry.func_names[i] << "," << perf << endl;
  }

  // Pick one joint_probs implementation to populate the variables.
  auto joint_probs = joint_probs_func_registry.funcs[0];
  for (int i = 0; i < grad_desc_func_registry.num_funcs; i++) {
    perf = perf_test_grad_desc(grad_desc_func_registry.funcs[i], joint_probs, X,
                               Y);
    cout << grad_desc_func_registry.func_names[i] << "," << perf << endl;
  }

  return 0;
}
