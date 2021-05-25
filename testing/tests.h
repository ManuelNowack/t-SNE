#include <string.h>
#include <tsne/matrix.h>
#include "base.h"

#define PRECISION_ERR 1e-6

/**
 * compares the n double values of the baseline and the modified function. Precision is the
 * tolerated error due to reordering of operations or similar.
 */
void compareData(double *orig, double *alt, int n, double precision);


void test_calc_squared_euclid_dist(void (*new_f)(Matrix *, Matrix *), Matrix *X, Matrix *D);

void test_calc_log_perplexity(void (*new_f)(double *, double *, int, int, double, double *, double *),
                        double* distances, double* probabilities, int n, int k,
                        double precision);

void test_joint_probs(void (*new_f)(Matrix *, Matrix *, Matrix *), Matrix *X, Matrix *P, Matrix *D);

void test_calc_affinities(void (*new_f)(Matrix *, Matrix *, Matrix *, Matrix *),
                            Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D);

void test_calc_cost(double (*new_f)(Matrix *, Matrix *), Matrix *P, Matrix *Q);

void test_grad_desc(void (*new_f)(Matrix *, tsne_var_t *, int, int, double),
                        Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum);

void test_tsne(void (*new_f)(Matrix *, Matrix *, tsne_var_t *, int),
                     Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim);