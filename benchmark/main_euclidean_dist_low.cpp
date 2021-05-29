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
#include <tsne/matrix.h>

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "benchmark.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 3 && argc != 5) {
    cerr << "Usage: " << argv[0] << " X_PCA Y_INIT [MIN MAX]" << endl;
    cerr << "If MIN and MAX are not defined, benchmark for full dataset." << endl;
    cerr << "Otherwise, benchmark from 2^MIN up to 2^MAX samples with multiplicative steps of 2." << endl;
    return 1;
  }

  Matrix X = load_matrix(argv[1]);
  Matrix Y = load_matrix(argv[2]);

  // Determine number of samples used for benchmarking
  int n_measurements;
  int log2_min_samples, log2_max_samples;
  if (argc == 3) {
    // All samples only
    n_measurements = 1;
  } else {
    // Different numbers of samples
    log2_min_samples = atoi(argv[3]);
    log2_max_samples = atoi(argv[4]);
    if (log2_min_samples > log2_max_samples || log2_min_samples < 0 || log2_max_samples < 0) {
      cerr << "Invalid sample bounds: MIN=" << log2_min_samples << ", MAX=" << log2_max_samples << endl;
      return 1;
    }
    if (X.nrows < powl(2, log2_max_samples)) {
      cerr << "Maximum number of samples (2^" << log2_max_samples
      << " = " << (int)powl(2, log2_max_samples)
      << ") is higher than the number of samples in the dataset provided ("
      << X.nrows << ")" << endl;
      return 1;
    }
    n_measurements = log2_max_samples - log2_min_samples + 1;
  }

  // Create array of sample numbers
  int n_samples_values[n_measurements];
  if (argc == 3) {
    n_samples_values[0] = X.nrows;
  } else {
    for (int i=0; i<n_measurements; i++) {
      n_samples_values[i] = (int)powl(2, log2_min_samples+i);
    }
  }

  register_functions();
  auto &euclidean_dist_func_registry =
      FuncRegistry<euclidean_dist_func_t>::get_instance();

  // TODO(mrettenba): Check validity of functions.

  int n_measurement_series = euclidean_dist_func_registry.num_funcs; 
  double performances[n_measurements][n_measurement_series];

  for (int i_measurement=0; i_measurement<n_measurements; i_measurement++) {
    cout << n_samples_values[i_measurement] << " samples" << endl;

    int n_samples = n_samples_values[i_measurement];
    Matrix Y_sub = {.nrows = n_samples, .ncols = Y.ncols, .data = Y.data};

    int i_series = 0;

    double perf;
    // Benchmark euclidean_dist functions
    for (int i = 0; i < euclidean_dist_func_registry.num_funcs; i++) {
      perf = perf_test_euclidean_dist(euclidean_dist_func_registry.funcs[i], Y_sub);
      cout << euclidean_dist_func_registry.func_names[i] << "," << perf << endl;
      performances[i_measurement][i_series] = perf;
      i_series++;
    }
    
    cout << endl;
  }
  
  for (int i = 0; i < euclidean_dist_func_registry.num_funcs; i++) {
    cout << euclidean_dist_func_registry.func_names[i];
    if (i == euclidean_dist_func_registry.num_funcs-1) {
      cout << endl;
    } else {
      cout << ", ";
    }
  }

  for (int i=0; i<n_measurements; i++) {
    for (int j=0; j<n_measurement_series; j++) {
      cout << performances[i][j];
      if (j == n_measurement_series-1) {
        cout << endl;
      } else {
        cout << ", ";
      }
    }
  }

  return 0;
}
