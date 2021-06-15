#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tsne/debug.h>
#include <tsne/matrix.h>


void create_tsne_variables(tsne_var_t &var, int n, int m) {
  var.P = create_matrix(n, n);
  var.Q = create_matrix(n, n);
  var.Q_numerators = create_matrix(n, n);
  var.grad_Y = create_matrix(n, m);
  var.Y_delta = create_matrix(n, m);
  var.tmp = create_matrix(n, n);
  var.gains = create_matrix(n, m);
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


/*
 * Load data in text file at filepath into Matrix structure.
 */
Matrix load_matrix(const char *filepath) {
  // open file
  FILE *in_file = fopen(filepath, "r");
  if (in_file == NULL) {
    throw std::runtime_error(
        std::string("Could not load matrix from filepath ") + filepath);
  }

  // determine matrix dimension
  Matrix A = {.nrows = 0, .ncols = 0, .data = NULL};

  double val;
  char character;

  // count columns
  do {
    fscanf(in_file, "%lf", &val);
    A.ncols++;
    character = (char)fgetc(in_file);
  } while (character != '\n' && character != EOF);
  A.nrows++;

  // count rows
  int counter = 0;
  do {
    fscanf(in_file, "%lf", &val);
    counter++;
    character = (char)fgetc(in_file);
    if (character == '\n') {
      // check if row contains as many elements as required
      if (counter != A.ncols) {
        throw std::runtime_error(
            std::string("Error: Invalid number of elements in row ") +
            std::to_string(A.nrows) + ": " + std::to_string(counter) +
            " instad of " + std::to_string(A.ncols));
      }
      counter = 0;
      A.nrows++;
    }
  } while (character != EOF);

  // load data
  A.data = (double *)aligned_alloc(32, A.nrows * A.ncols * sizeof(double));
  if (!A.data) {
    throw std::runtime_error("Could not allocate memory to store matrix.");
  }
  rewind(in_file);
  int i = 0;
  do {
    fscanf(in_file, "%lf", &A.data[i]);
    i++;
    character = (char)fgetc(in_file);
  } while (character != EOF);
  fclose(in_file);

  DEBUG("Matrix of dimension " << A.nrows << " x " << A.ncols << " loaded");

  return A;
}

/*
 * Store matrix A into a text file at filepath.
 */
void store_matrix(const char *filepath, Matrix A) {
  FILE *out_file = fopen(filepath, "w");

  if (out_file == NULL) {
    throw std::runtime_error(
        std::string("Could not store matrix to filepath ") + filepath);
  }

  int n = A.nrows;
  int m = A.ncols;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m - 1; j++) {
      fprintf(out_file, "%.18e ", A.data[m * i + j]);
    }
    fprintf(out_file, "%.18e\n", A.data[m * i + m - 1]);
  }

  fclose(out_file);
}

Matrix create_matrix(int nrows, int ncols) {
  Matrix A = {
      .nrows = nrows,
      .ncols = ncols,
      .data = (double *)aligned_alloc(32, nrows * ncols * sizeof(double))};
  memset(A.data, 0, nrows * ncols * sizeof(double));
  if (!A.data) {
    throw std::runtime_error("Could not allocate memory for matrix.");
  }

  return A;
}

void copy_matrix(const Matrix *orig, Matrix *copy) {
  copy->ncols = orig->ncols;
  copy->nrows = orig->nrows;
  size_t datasize = copy->nrows * copy->ncols * sizeof(double);
  copy->data = (double *)aligned_alloc(32, datasize);
  memcpy(copy->data, orig->data, datasize);
  if (!copy->data) {
    throw std::runtime_error("Could not allocate memory for matrix.");
  }
}
