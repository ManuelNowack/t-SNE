#ifndef TSNE_TEST_H_
#define TSNE_TEST_H_
#include <gtest/gtest.h>
#include <tsne/func_registry.h>
#include <tsne/matrix.h>

#define PRECISION_ERR (1e-3)

class BaseTest : public testing::Test {
 protected:
  void SetUp() override {
    load_matrix("mnist100_X_pca.txt", &X);
    load_matrix("mnist100_Y_init.txt", &Y_expected);
    copy_matrix(&Y_expected, &Y_actual);

    n = X.nrows;
    n_dim = Y_expected.ncols;

    create_tsne_variables(var_expected, n, n_dim);
    create_tsne_variables(var_actual, n, n_dim);
  }

  void TearDown() override {
    destroy_tsne_variables(var_expected);
    destroy_tsne_variables(var_actual);

    free(X.data);
    free(Y_expected.data);
    free(Y_actual.data);
  }

  int n, n_dim;
  Matrix X, Y_expected, Y_actual;
  tsne_var_t var_expected, var_actual;
};

class TsneTest : public BaseTest,
                 public testing::WithParamInterface<tsne_func_t *> {};
class JointProbsTest
    : public BaseTest,
      public testing::WithParamInterface<joint_probs_func_t *> {};
class GradDescTest : public BaseTest,
                     public testing::WithParamInterface<grad_desc_func_t *> {};
class LogPerplexityTest
    : public BaseTest,
      public testing::WithParamInterface<log_perplexity_func_t *> {};

class EuclideanDistTest : public BaseTest,
                          public testing::WithParamInterface<euclidean_dist_func_t *> {};

class EuclideanDistLowTest : public BaseTest,
                             public testing::WithParamInterface<euclidean_dist_func_t *> {};

#endif  // TSNE_TEST_H_
