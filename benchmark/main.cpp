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
#define REP 50

// Computes and reports the number of cycles required per iteration.
// for the given function
double perf_test(tsne_func_t *f, string desc) {
  double cycles = 0.;
  long num_runs = 100;
  double multiplier = 1;
  uint64_t start, end;

  Matrix X = load_matrix("mnist2500_X_pca.txt");
  Matrix Y = load_matrix("mnist2500_Y_init.txt");

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  /*
  do {
      num_runs = num_runs * multiplier;
      start = start_tsc();
      for (size_t i = 0; i < num_runs; i++) {
          f(X, 2, 20, Y);
      }
      end = stop_tsc(start);

      cycles = (double)end;
      multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);
  */

  // Actual performance measurements repeated REP times.
  // We simply store all results and compute medians during post-processing.
  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {
    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(X, 2, 20, Y);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;
    total_cycles += cycles;
  }
  total_cycles /= REP;

  cycles = total_cycles;

  return cycles;
}

int main(int argc, char **argv) {
  double perf;
  int i;

  register_functions();
  FuncResitry &func_registry = FuncResitry::get_instance();

  cout << func_registry.num_funcs << " functions registered." << endl;

  // TODO(mrettenba): Check validity of functions.

  for (i = 0; i < func_registry.num_funcs; i++) {
    perf = perf_test(func_registry.funcs[i], func_registry.func_names[i]);
    cout << func_registry.func_names[i] << "," << perf << endl;
  }

  return 0;
}