#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  int nrows, ncols;
  double *data;  // matrix elements in row major order
} Matrix;

// Intermediate variables used in t-SNE calculations.
typedef struct {
  Matrix P;
  Matrix Q;
  Matrix Q_numerators;
  Matrix grad_Y;
  Matrix Y_delta;
  Matrix tmp;
  Matrix gains;
  Matrix D;
} tsne_var_t;

Matrix load_matrix(const char *filepath);
void store_matrix(const char *filepath, Matrix A);
Matrix create_matrix(int nrows, int ncols);
void copy_matrix(Matrix *orig, Matrix *copy);
void assert_finite_matrix(Matrix A);

#endif
