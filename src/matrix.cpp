#include <assert.h>
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
 * Return value:
 *  0 Matrix was successfully loaded
 *  1 Failed to open file at filepath
 *  2 Failed to read next double in the file
 *  3 Unexpected end of file
 *  4 Unexpected number of doubles in a row 
 *  5 Failed to allocate matrix
 */
int load_matrix_verbose(const char *filepath, Matrix *A) {
  FILE *in_file = fopen(filepath, "r");
  if (!in_file) {
    return 1;
  }

  double val;
  int delimiter, items_read;

  // count columns
  A->ncols = 0;
  do {
    items_read = fscanf(in_file, "%lf", &val);
    if (items_read != 1) {
      return 2;
    }
    A->ncols++;
    delimiter = fgetc(in_file);
    if (delimiter == EOF) {
      return 3;
    }
  } while (delimiter != '\n');
  A->nrows = 1;

  // count rows
  int counter = 0;
  do {
    items_read = fscanf(in_file, "%lf", &val);
    if (items_read == EOF) {
      break;
    }
    if (items_read != 1) {
      return 2;
    }
    counter++;
    delimiter = fgetc(in_file);
    if (delimiter == '\n') {
      // check if row contains as many elements as required
      if (counter != A->ncols) {
        return 4;
      }
      counter = 0;
      A->nrows++;
    }
  } while (delimiter != EOF);

  // load data
  A->data = (double *)aligned_alloc(32, A->nrows * A->ncols * sizeof(double));
  if (!A->data) {
    return 5;
  }
  rewind(in_file);
  int i = 0;
  do {
    items_read = fscanf(in_file, "%lf", &A->data[i++]);
    if (items_read == EOF) {
      break;
    }
    if (items_read != 1) {
      return 2;
    }
    delimiter = fgetc(in_file);
  } while (delimiter != EOF);
  fclose(in_file);

  return 0;
}

/*
 * Load data in text file at filepath into Matrix structure.
 */
void load_matrix(const char *filepath, Matrix *A) {
  assert(load_matrix_verbose(filepath, A) == 0);
  DEBUG("Matrix of dimension " << A->nrows << " x " << A->ncols << " loaded");
}

/*
 * Store matrix A into a text file at filepath.
 */
void store_matrix(const char *filepath, Matrix *A) {
  FILE *out_file = fopen(filepath, "w");

  if (out_file == NULL) {
    throw std::runtime_error(
        std::string("Could not store matrix to filepath ") + filepath);
  }

  int n = A->nrows;
  int m = A->ncols;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m - 1; j++) {
      fprintf(out_file, "%.18e ", A->data[m * i + j]);
    }
    fprintf(out_file, "%.18e\n", A->data[m * i + m - 1]);
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
