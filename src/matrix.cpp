#include <stdio.h>
#include <stdlib.h>
#include <tsne/matrix.h>

Matrix load_matrix(const char *filepath) {
  /*
   * Load data in text file at filepath into Matrix structure.
   */

  // open file
  FILE *in_file = fopen(filepath, "r");
  if (in_file == NULL) {
    printf("Could not load matrix from filepath %s\n", filepath);
    exit(-1);
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
        printf(
            "Error: Invalid number of elements in row %d. %d instad of %d.\n",
            A.nrows, counter, A.ncols);
        exit(-1);
      }
      counter = 0;
      A.nrows++;
    }
  } while (character != EOF);

  // load data
  A.data = (double *)malloc(A.nrows * A.ncols * sizeof(double));
  if (!A.data) {
    printf("Error: Could not allocate memory to store matrix.\n");
    exit(-1);
  }
  rewind(in_file);
  int i = 0;
  do {
    fscanf(in_file, "%lf", &A.data[i]);
    i++;
    character = (char)fgetc(in_file);
  } while (character != EOF);
  fclose(in_file);

  printf("Matrix of dimension %d x %d loaded.\n", A.nrows, A.ncols);

  return A;
}

void store_matrix(const char *filepath, Matrix A) {
  /*
   * Store matrix A into a text file at filepath.
   */

  FILE *out_file = fopen(filepath, "w");

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
  Matrix A = {.nrows = nrows,
              .ncols = ncols,
              .data = (double *)malloc(nrows * ncols * sizeof(double))};
  if (!A.data) {
    printf("Error: Could not allocate memory for matrix.\n");
    exit(-1);
  }

  return A;
}

void elementwise_matrix_subtraction(Matrix op1, Matrix op2, Matrix res) {
  /*
   * Calculate elementwise matrix difference res = op1 - op2
   */

  int n = res.nrows;
  int m = res.ncols;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      res.data[i * m + j] = op1.data[i * m + j] - op2.data[i * m + j];
    }
  }
}

void elementwise_matrix_multiplication(Matrix op1, Matrix op2, Matrix res) {
  /*
   * Calculate elementwise matrix multiplication res = op1 * op2
   */

  int n = res.nrows;
  int m = res.ncols;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      res.data[i * m + j] = op1.data[i * m + j] * op2.data[i * m + j];
    }
  }
}

void assert_finite_matrix(Matrix A) {
  printf("assert_finite_matrix not implemented.\n");
  exit(-1);
}
