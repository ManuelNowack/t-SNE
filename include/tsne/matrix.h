#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  int nrows, ncols;
  double *data;  // matrix elements in row major order
} Matrix;

Matrix load_matrix(const char *filepath);
void store_matrix(const char *filepath, Matrix A);
Matrix create_matrix(int nrows, int ncols);
void assert_finite_matrix(Matrix A);

#endif