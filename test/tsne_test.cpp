#include "tsne_test.h"

#include <gtest/gtest.h>
#include <math.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

// Put the functions you want to test here.
INSTANTIATE_TEST_SUITE_P(Tsne, JointProbsTest,
                         testing::Values(&joint_probs_baseline,
                                         &joint_probs_unroll8,
                                         &joint_probs_avx_fma_acc4));

INSTANTIATE_TEST_SUITE_P(Tsne, GradDescTest,
                         testing::Values(&grad_desc_baseline,
                                         &grad_desc_no_vars_baseline,
                                         &grad_desc_no_vars_tmp,
                                         &grad_desc_no_vars_D,
                                         &grad_desc_no_vars_Q,
                                         &grad_desc_no_vars_Q_numerators,
                                         &grad_desc_no_vars_scalar,
                                         &grad_desc_no_vars_no_if,
                                         &grad_desc_no_vars_grad,
                                         &grad_desc_no_vars_means,
                                         &grad_desc_no_vars_unroll2,
                                         &grad_desc_no_vars_unroll4,
                                         &grad_desc_no_vars_unroll6,
                                         &grad_desc_no_vars_unroll8,
                                         &grad_desc_no_vars_fetch,
                                         &grad_desc_no_vars_no_l,
                                         &grad_desc_no_vars_unroll,
                                         &grad_desc_no_vars_vector,
                                         &grad_desc_no_vars_vector_acc,
                                         &grad_desc_no_vars_vector_inner,
                                         &grad_desc_no_vars_vector_unroll2));

INSTANTIATE_TEST_SUITE_P(
    Tsne, LogPerplexityTest,
    testing::Values(&log_perplexity_baseline, &log_perplexity_unroll2,
                    &log_perplexity_unroll4, &log_perplexity_unroll8,
                    &log_perplexity_avx, &log_perplexity_avx_acc4,
                    &log_perplexity_avx_fma_acc4));

INSTANTIATE_TEST_SUITE_P(Tsne, TsneTest,
                         testing::Values(&tsne_baseline,
                                         &tsne_no_vars,
                                         &tsne_scalar,
                                         &tsne_vec));

INSTANTIATE_TEST_SUITE_P(Tsne, EuclideanDistTest,
                         testing::Values(&euclidean_dist_baseline,
                                         &euclidean_dist_alt_unroll4,
                                         &euclidean_dist_alt_vec_unroll4x4));
INSTANTIATE_TEST_SUITE_P(Tsne, EuclideanDistLowTest,
                         testing::Values(&euclidean_dist_baseline,
                                         &euclidean_dist_low_unroll,
                                         &euclidean_dist_low_vec3_unroll4x8));

// compares the n double values of the baseline and the modified function.
// Precision is the tolerated error due to reordering of operations or similar.
testing::AssertionResult IsArrayNear(double *expected, double *actual, int n,
                                     const char *name,
                                     double precision = PRECISION_ERR) {
  double diff = 0;
  for (int i = 0; i < n; i++) {
    if (expected[i] == HUGE_VAL || actual[i] == HUGE_VAL) {
      if (actual[i] != HUGE_VAL || expected[i] != HUGE_VAL) {
        // case where we don't have two HUGE_VALS
        return testing::AssertionFailure()
               << "Encountered HUGE_VAL and normal value mismatch, data not "
                  "equal for"
               << name << ": expected: " << expected[i]
               << " actual:" << actual[i];
      }
    } else {
      diff += abs(expected[i] - actual[i]);
    }
  }
  if (diff > precision) {
    return testing::AssertionFailure()
           << "Absolute diff of " << name << ": " << diff << "is larger than "
           << precision;
  }
  return testing::AssertionSuccess();
}

// Compares upper triangular values of the baseline and the modified function.
// Precision is the tolerated error due to reordering of operations or similar.
testing::AssertionResult IsUpperTriangleNear(double *expected, double *actual,
                                             int nrows, int ncols,
                                             const char *name,
                                             double precision = PRECISION_ERR) {
  double diff = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = i + 1; j < ncols; j++) {
      if (expected[ncols * i + j] == HUGE_VAL ||
          actual[ncols * i + j] == HUGE_VAL) {
        if (actual[ncols * i + j] != HUGE_VAL ||
            expected[ncols * i + j] != HUGE_VAL) {
          // case where we don't have two HUGE_VALS
          return testing::AssertionFailure()
                 << "Encountered HUGE_VAL and normal value mismatch, data not "
                    "equal for"
                 << name << ": expected: " << expected[ncols * i + j]
                 << " actual:" << actual[ncols * i + j];
        }
      } else {
        diff += abs(expected[ncols * i + j] - actual[ncols * i + j]);
      }
    }
  }
  if (diff > precision) {
    return testing::AssertionFailure()
           << "Absolute diff of " << name << ": " << diff << "is larger than "
           << precision;
  }
  return testing::AssertionSuccess();
}

void compare_tsne_var(tsne_var_t &expected, tsne_var_t &actual,
                      double precision = PRECISION_ERR) {
  EXPECT_TRUE(IsArrayNear(expected.D.data, actual.D.data,
                          expected.D.ncols * expected.D.nrows, "D", precision));
  EXPECT_TRUE(IsArrayNear(expected.gains.data, actual.gains.data,
                          expected.gains.ncols * expected.gains.nrows, "gains",
                          precision));
  EXPECT_TRUE(IsArrayNear(expected.grad_Y.data, actual.grad_Y.data,
                          expected.grad_Y.ncols * expected.grad_Y.nrows,
                          "grad_Y", precision));
  EXPECT_TRUE(IsArrayNear(expected.P.data, actual.P.data,
                          expected.P.ncols * expected.P.nrows, "P", precision));
  EXPECT_TRUE(IsArrayNear(expected.Q.data, actual.Q.data,
                          expected.Q.ncols * expected.Q.nrows, "Q", precision));
  EXPECT_TRUE(
      IsArrayNear(expected.Q_numerators.data, actual.Q_numerators.data,
                  expected.Q_numerators.ncols * expected.Q_numerators.nrows,
                  "Q_numerator", precision));
  EXPECT_TRUE(IsArrayNear(expected.tmp.data, actual.tmp.data,
                          expected.tmp.ncols * expected.tmp.nrows, "tmp",
                          precision));
  EXPECT_TRUE(IsArrayNear(expected.Y_delta.data, actual.Y_delta.data,
                          expected.Y_delta.ncols * expected.Y_delta.nrows,
                          "Y_delta", precision));
}

TEST_P(JointProbsTest, IsValid) {
  joint_probs_baseline(&X, &var_expected.P, &var_expected.D);
  GetParam()(&X, &var_actual.P, &var_actual.D);

  EXPECT_TRUE(IsArrayNear(X.data, X.data, X.ncols * X.nrows, "X"));
  EXPECT_TRUE(IsArrayNear(var_expected.D.data, var_actual.D.data,
                          var_expected.D.ncols * var_expected.D.nrows, "D"));
  EXPECT_TRUE(IsArrayNear(var_expected.P.data, var_actual.P.data,
                          var_expected.P.ncols * var_expected.P.nrows, "P"));
}

TEST_P(GradDescTest, IsValid) {
  joint_probs_baseline(&X, &var_expected.P, &var_expected.D);
  grad_desc_baseline(&Y_expected, &var_expected, n, n_dim, kFinalMomentum);
  joint_probs_baseline(&X, &var_actual.P, &var_actual.D);
  GetParam()(&Y_actual, &var_actual, n, n_dim, kFinalMomentum);

  EXPECT_TRUE(IsArrayNear(Y_expected.data, Y_actual.data,
                          Y_expected.ncols * Y_expected.nrows, "Y", 0.0));
  // compare_tsne_var(var_expected, var_actual, 0.0);
}

TEST_P(LogPerplexityTest, IsValid) {
  euclidean_dist_baseline(&X, &var_expected.D);
  double *distances = var_expected.D.data;
  double *p_expected = var_expected.P.data;
  double *p_actual = var_actual.P.data;
  double log_perp_expected, log_perp_actual;
  double norm_expected, norm_actual;

  log_perplexity_baseline(distances, p_expected, n, 0, 0.5, &log_perp_expected,
                          &norm_expected);
  GetParam()(distances, p_actual, n, 0, 0.5, &log_perp_actual, &norm_actual);

  EXPECT_TRUE(IsArrayNear(p_expected, p_actual, n, "probabilities"));
  EXPECT_NEAR(log_perp_expected, log_perp_actual, PRECISION_ERR);
  EXPECT_NEAR(norm_expected, norm_actual, PRECISION_ERR);
}

TEST_P(TsneTest, IsValid) {
  tsne_baseline(&X, &Y_expected, &var_expected, n_dim);
  GetParam()(&X, &Y_actual, &var_actual, n_dim);

  EXPECT_TRUE(IsArrayNear(X.data, X.data, X.ncols * X.nrows, "X"));
  EXPECT_TRUE(IsArrayNear(Y_expected.data, Y_actual.data,
                          Y_expected.ncols * Y_expected.nrows, "Y"));
  // compare_tsne_var(var_expected, var_actual);
}

TEST_P(EuclideanDistTest, IsValid) {
  euclidean_dist_baseline(&X, &var_expected.D);
  GetParam()(&X, &var_actual.D);

  EXPECT_TRUE(IsArrayNear(X.data, X.data, X.ncols * X.nrows, "X"));
  compare_tsne_var(var_expected, var_actual);
}

TEST_P(EuclideanDistLowTest, IsValid) {
  euclidean_dist_baseline(&Y_expected, &var_expected.D);
  GetParam()(&Y_actual, &var_actual.D);

  EXPECT_TRUE(IsArrayNear(Y_expected.data, Y_actual.data,
                          Y_expected.ncols * Y_expected.nrows, "Y"));
  EXPECT_TRUE(IsUpperTriangleNear(var_expected.D.data, var_actual.D.data,
                                  var_expected.D.nrows, var_expected.D.ncols,
                                  "D"));
}

/*
void test_calc_affinities(void (*new_f)(Matrix *, Matrix *, Matrix *, Matrix *),
                          Matrix *Y, Matrix *Q, Matrix *Q_numerators,
                          Matrix *D) {
  printf("Testing calc_affinities:\n");
  Matrix Yn, Qn, Q_num, Dn;
  copy_matrix(Y, &Yn);
  copy_matrix(Q, &Qn);
  copy_matrix(Q_numerators, &Q_num);
  copy_matrix(D, &Dn);

  calc_affinities(Y, Q, Q_numerators, D);
  new_f(&Yn, &Qn, &Q_num, &Dn);

  IsArrayNear(Y.data, Yn.data, Y.ncols * Y.nrows, "Y");
  IsArrayNear(Q.data, Qn.data, Q.ncols * Q.nrows, PRECISION_ERR, "Q");
  IsArrayNear(Q_numerators.data, Q_num.data,
              Q_numerators.ncols * Q_numerators.nrows, "Q_numerators");
  IsArrayNear(D.data, Dn.data, D.ncols * D.nrows, "D");

  free((void *)Yn.data);
  free((void *)Qn.data);
  free((void *)Q_num.data);
  free((void *)Dn.data);
}

void test_calc_cost(double (*new_f)(Matrix *, Matrix *), Matrix *P, Matrix *Q) {
  printf("Testing calc_cost:\n");
  double res1, res2;

  res1 = calc_cost(P, Q);
  res2 = new_f(P, Q);

  IsArrayNear(&res1, &res2, 1, PRECISION_ERR, "cost");
}
*/
