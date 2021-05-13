#ifndef TSNE_TEST_H_
#define TSNE_TEST_H_
#include <gtest/gtest.h>
#include <tsne/func_registry.h>
#include <tsne/matrix.h>

#define PRECISION_ERR (1e-3)

class BaseTest : public testing::Test {
 protected:
  void SetUp() override {
    X = load_matrix("mnist100_X_pca.txt");
    Y_expected = load_matrix("mnist100_Y_init.txt");
    copy_matrix(&Y_expected, &Y_actual);

    n = X.nrows;
    n_dim = Y_expected.ncols;

    _create_tsne_variables(var_expected, n, n_dim);
    _create_tsne_variables(var_actual, n, n_dim);
  }

  void TearDown() override {
    _destroy_tsne_variables(var_expected);
    _destroy_tsne_variables(var_actual);

    free(X.data);
    free(Y_expected.data);
    free(Y_actual.data);
  }

  int n, n_dim;
  Matrix X, Y_expected, Y_actual;
  tsne_var_t var_expected, var_actual;

 private:
  void _create_tsne_variables(tsne_var_t &var, int n, int n_dim) {
    var.P = create_matrix(n, n);
    var.Q = create_matrix(n, n);
    var.Q_numerators = create_matrix(n, n);
    var.grad_Y = create_matrix(n, n_dim);
    var.Y_delta = create_matrix(n, n_dim);
    var.tmp = create_matrix(n, n);
    var.gains = create_matrix(n, n_dim);
    var.D = create_matrix(n, n);
  }

  void _destroy_tsne_variables(tsne_var_t &var) {
    free(var.P.data);
    free(var.Q.data);
    free(var.Q_numerators.data);
    free(var.grad_Y.data);
    free(var.Y_delta.data);
    free(var.tmp.data);
    free(var.gains.data);
    free(var.D.data);
  }
};

class TsneTest : public BaseTest,
                 public testing::WithParamInterface<tsne_func_t *> {};
class JointProbsTest
    : public BaseTest,
      public testing::WithParamInterface<joint_probs_func_t *> {};
class GradDescTest : public BaseTest,
                     public testing::WithParamInterface<grad_desc_func_t *> {};

#endif  // TSNE_TEST_H_
