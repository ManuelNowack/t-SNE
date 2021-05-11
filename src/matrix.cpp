#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tsne/debug.h>
#include <tsne/matrix.h>

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
  A.data = (double *)malloc(A.nrows * A.ncols * sizeof(double));
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
  Matrix A = {.nrows = nrows,
              .ncols = ncols,
              .data = (double *)malloc(nrows * ncols * sizeof(double))};
  if (!A.data) {
    throw std::runtime_error("Could not allocate memory for matrix.");
  }

  return A;
}

void assert_finite_matrix(Matrix A) {
  throw std::runtime_error("assert_finite_matrix not implemented.");
}

void copy_matrix(Matrix *orig, Matrix *copy){
  copy->ncols = orig->ncols;
  copy->nrows = orig->nrows;
  size_t datasize = copy->nrows*copy->ncols*sizeof(double);
  copy->data = (double *)malloc(datasize);
  memcpy(copy->data, orig->data, datasize);
  if (!copy->data) {
    throw std::runtime_error("Could not allocate memory for matrix.");
  }
}
