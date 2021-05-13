#include "tsne_test.h"

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <math.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

// Put the functions you want to test here.
joint_probs_func_t joint_probs_baseline;
INSTANTIATE_TEST_SUITE_P(Tsne, JointProbsTest,
                         testing::Values(&joint_probs_baseline));

grad_desc_func_t grad_desc_baseline;
INSTANTIATE_TEST_SUITE_P(Tsne, GradDescTest,
                         testing::Values(&grad_desc_baseline));

tsne_func_t tsne_baseline;
INSTANTIATE_TEST_SUITE_P(Tsne, TsneTest, testing::Values(&tsne_baseline));

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

void compare_tsne_var(tsne_var_t &expected, tsne_var_t &actual) {
  EXPECT_TRUE(IsArrayNear(expected.D.data, actual.D.data,
                          expected.D.ncols * expected.D.nrows, "D"));
  EXPECT_TRUE(IsArrayNear(expected.gains.data, actual.gains.data,
                          expected.gains.ncols * expected.gains.nrows,
                          "gains"));
  EXPECT_TRUE(IsArrayNear(expected.grad_Y.data, actual.grad_Y.data,
                          expected.grad_Y.ncols * expected.grad_Y.nrows,
                          "grad_Y"));
  EXPECT_TRUE(IsArrayNear(expected.P.data, actual.P.data,
                          expected.P.ncols * expected.P.nrows, "P"));
  EXPECT_TRUE(IsArrayNear(expected.Q.data, actual.Q.data,
                          expected.Q.ncols * expected.Q.nrows, "Q"));
  EXPECT_TRUE(
      IsArrayNear(expected.Q_numerators.data, actual.Q_numerators.data,
                  expected.Q_numerators.ncols * expected.Q_numerators.nrows,
                  "Q_numerator"));
  EXPECT_TRUE(IsArrayNear(expected.tmp.data, actual.tmp.data,
                          expected.tmp.ncols * expected.tmp.nrows, "tmp"));
  EXPECT_TRUE(IsArrayNear(expected.Y_delta.data, actual.Y_delta.data,
                          expected.Y_delta.ncols * expected.Y_delta.nrows,
                          "Y_delta"));
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
  grad_desc_baseline(&Y_expected, &var_expected, n, n_dim, kFinalMomentum);
  GetParam()(&Y_actual, &var_actual, n, n_dim, kFinalMomentum);

  EXPECT_TRUE(IsArrayNear(Y_expected.data, Y_actual.data,
                          Y_expected.ncols * Y_expected.nrows, "Y"));
  compare_tsne_var(var_expected, var_actual);
}

TEST_P(TsneTest, IsValid) {
  tsne_baseline(&X, &Y_expected, &var_expected, n_dim);
  GetParam()(&X, &Y_actual, &var_actual, n_dim);

  EXPECT_TRUE(IsArrayNear(X.data, X.data, X.ncols * X.nrows, "X"));
  EXPECT_TRUE(IsArrayNear(Y_expected.data, Y_actual.data,
                          Y_expected.ncols * Y_expected.nrows, "Y"));
  compare_tsne_var(var_expected, var_actual);
}

/*
// Tests to be refactored.
void test_calc_squared_euclid_dist(void (*new_f)(Matrix *, Matrix *), Matrix *X,
                                   Matrix *D) {
  printf("Testing calc_squared_euclid_distance:\n");
  Matrix D_new, X_new;
  copy_matrix(D, &D_new);
  copy_matrix(X, &X_new);

  euclidean_dist_baseline(X, D);
  new_f(&X_new, &D_new);

  IsArrayNear(X.data, X_new.data, X.ncols * X.nrows, "X");
  IsArrayNear(D.data, D_new.data, D.ncols * D.nrows, "D");

  free((void *)D_new.data);
  free((void *)X_new.data);
}

void test_calc_log_perplexity(void (*new_f)(double *, double *, int, int,
                                            double, double *, double *),
                              double *distances, double *probabilities, int n,
                              int k, double precision) {
  printf("Testing calc_log_perplexity:\n");
  double *prob_new = (double *)malloc(n * sizeof(double));
  memcpy(prob_new, probabilities, n * sizeof(double));
  double log_perp1, log_perp2, normalizer1, normalizer2;

  calc_log_perplexity(distances, probabilities, n, k, precision, &log_perp1,
                      &normalizer1);
  new_f(distances, prob_new, n, k, precision, &log_perp2, &normalizer2);

  IsArrayNear(probabilities, prob_new, n, "probabilities");
  IsArrayNear(&log_perp1, &log_perp2, 1, "log_perp");
  IsArrayNear(&normalizer1, &normalizer2, 1, "normalizer");

  free((void *)prob_new);
}

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
