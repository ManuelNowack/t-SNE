// Joint probs related benchmarks.

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
    cerr << "If MIN and MAX are not defined, benchmark for full dataset."
         << endl;
    cerr << "Otherwise, benchmark from 2^MIN up to 2^MAX samples with "
            "multiplicative steps of 2."
         << endl;
    return 1;
  }

  Matrix X, Y;
  load_matrix(argv[1], &X);
  load_matrix(argv[2], &Y);

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
    if (log2_min_samples > log2_max_samples || log2_min_samples < 0 ||
        log2_max_samples < 0) {
      cerr << "Invalid sample bounds: MIN=" << log2_min_samples
           << ", MAX=" << log2_max_samples << endl;
      return 1;
    }
    if (X.nrows < powl(2, log2_max_samples)) {
      cerr << "Maximum number of samples (2^" << log2_max_samples << " = "
           << (int)powl(2, log2_max_samples)
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
    for (int i = 0; i < n_measurements; i++) {
      n_samples_values[i] = (int)powl(2, log2_min_samples + i);
    }
  }

  register_functions();
  auto &log_perplexity_func_registry =
      FuncRegistry<log_perplexity_func_t>::get_instance();
  auto &joint_probs_func_registry =
      FuncRegistry<joint_probs_func_t>::get_instance();

  int n_measurement_series = log_perplexity_func_registry.num_funcs +
                             joint_probs_func_registry.num_funcs;
  double performances[n_measurements][n_measurement_series];

  for (int i_measurement = 0; i_measurement < n_measurements; i_measurement++) {
    cout << n_samples_values[i_measurement] << " samples" << endl;

    int n_samples = n_samples_values[i_measurement];
    Matrix X_sub = {.nrows = n_samples, .ncols = X.ncols, .data = X.data};
    Matrix Y_sub = {.nrows = n_samples, .ncols = Y.ncols, .data = Y.data};

    int i_series = 0;

    double perf;
    for (int i = 0; i < log_perplexity_func_registry.num_funcs; i++) {
      perf = perf_test_log_perplexity(log_perplexity_func_registry.funcs[i],
                                      X_sub);
      cout << log_perplexity_func_registry.func_names[i] << "," << perf << endl;
      performances[i_measurement][i_series] = perf;
      i_series++;
    }

    for (int i = 0; i < joint_probs_func_registry.num_funcs; i++) {
      perf = perf_test_joint_probs(joint_probs_func_registry.funcs[i], X_sub);
      cout << joint_probs_func_registry.func_names[i] << "," << perf << endl;
      performances[i_measurement][i_series] = perf;
      i_series++;
    }

    cout << endl;
  }

  for (int i = 0; i < log_perplexity_func_registry.num_funcs; i++) {
    cout << log_perplexity_func_registry.func_names[i] << ", ";
  }

  for (int i = 0; i < joint_probs_func_registry.num_funcs; i++) {
    cout << joint_probs_func_registry.func_names[i];
    if (i == joint_probs_func_registry.num_funcs - 1) {
      cout << endl;
    } else {
      cout << ", ";
    }
  }

  for (int i = 0; i < n_measurements; i++) {
    for (int j = 0; j < n_measurement_series; j++) {
      cout << performances[i][j];
      if (j == n_measurement_series - 1) {
        cout << endl;
      } else {
        cout << ", ";
      }
    }
  }

  return 0;
}
