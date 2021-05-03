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

#include <tsne/benchmark.h>
#include <tsne/func_registry.h>
#include <tsne/matrix.h>

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tsc_x86.h"

using namespace std;

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
