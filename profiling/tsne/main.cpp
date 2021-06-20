#include <tsne/func_registry.h>
#include <tsne/matrix.h>

#include <iostream>
#include <random>

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

  const int num_funcs = 6;
  tsne_func_t *funcs[num_funcs] = {&tsne_baseline, &tsne_scalar, &tsne_vec,
                                   &tsne_vec2,     &tsne_vec3,   &tsne_no_vars};

  const int n_dim = 2;

  for (int i_measurement = 0; i_measurement < n_measurements; i_measurement++) {
    cout << n_samples_values[i_measurement] << " samples" << endl;

    int n_samples = n_samples_values[i_measurement];
    Matrix X_sub = {.nrows = n_samples, .ncols = X.ncols, .data = X.data};
    Matrix Y_sub = {.nrows = n_samples, .ncols = Y.ncols, .data = Y.data};
    tsne_var_t var;
    create_tsne_variables(var, n_samples, n_dim);

    for (int i = 0; i < num_funcs; i++) {
      funcs[i](&X_sub, &Y_sub, &var, n_dim);
    }

    destroy_tsne_variables(var);
  }

  return 0;
}
