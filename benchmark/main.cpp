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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tsne/func_registry.h>
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

// Computes and reports the number of cycles required per iteration.
// for the given tsne function.
double perf_test_tsne(tsne_func_t *f, string desc) {
  double cycles = 0.;
  long num_runs = 1;
  double multiplier = 1;
  uint64_t start, end;

  Matrix X = load_matrix("mnist2500_X_pca.txt");
  Matrix Y = load_matrix("mnist2500_Y_init.txt");
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

int main(int argc, char **argv) {
  double perf;
  int i;

  register_functions();
  auto &func_registry = FuncResitry<tsne_func_t>::get_instance();

  cout << func_registry.num_funcs << " functions registered." << endl;

  // TODO(mrettenba): Check validity of functions.

  for (i = 0; i < func_registry.num_funcs; i++) {
    perf = perf_test_tsne(func_registry.funcs[i], func_registry.func_names[i]);
    cout << func_registry.func_names[i] << "," << perf << endl;
  }

  return 0;
}
