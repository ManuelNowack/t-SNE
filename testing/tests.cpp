#include "tests.h"

/**
 * compares the n double values of the baseline and the modified function. Precision is the
 * tolerated error due to reordering of operations or similar.
 */
void compareData(double *orig, double *alt, int n, double precision, const char *name){
    double diff = 0;
    for(int i=0; i<n; i++){
        if(orig[i] == HUGE_VAL || alt[i] == HUGE_VAL){
            if(alt[i] != HUGE_VAL || orig[i] != HUGE_VAL){
                printf("Encountered HUGE_VAL and normal value mismatch, data not equal for %s: old: %lf, new: %lf",
                        name, orig[i], alt[i]);
                return; //case where we don't have two HUGE_VALS
            }
        }
        else{
            diff += abs(orig[i]-alt[i]);
        }
    }
    if(diff > precision){
        printf("Absolute diff of %s %lf is larger than %lf\n", name, diff, precision);
        return;
    }
    printf("Data of %s is equal.\n", name);
}


void test_calc_squared_euclid_dist(void (*new_f)(Matrix *, Matrix *), Matrix *X, Matrix *D){
    printf("Testing calc_squared_euclid_distance:\n");
    Matrix D_new, X_new;
    copy_matrix(D, &D_new);
    copy_matrix(X, &X_new);

    euclidean_dist_baseline(X, D);
    new_f(&X_new, &D_new);

    compareData(X->data, X_new.data, X->ncols*X->nrows, PRECISION_ERR, "X");
    compareData(D->data, D_new.data, D->ncols*D->nrows, PRECISION_ERR, "D");

    free((void *)D_new.data);
    free((void *)X_new.data);
}

void test_calc_log_perplexity(void (*new_f)(double *, double *, int, int, double, double *, double *),
                        double* distances, double* probabilities, int n, int k,
                        double precision) {
    printf("Testing calc_log_perplexity:\n");
    double *prob_new = (double *)malloc(n*sizeof(double));
    memcpy(prob_new, probabilities, n*sizeof(double));
    double log_perp1, log_perp2, normalizer1, normalizer2;

    calc_log_perplexity(distances, probabilities, n, k, precision, &log_perp1, &normalizer1);
    new_f(distances, prob_new, n, k, precision, &log_perp2, &normalizer2);

    compareData(probabilities, prob_new, n, PRECISION_ERR, "probabilities");
    compareData(&log_perp1, &log_perp2, 1, PRECISION_ERR, "log_perp");
    compareData(&normalizer1, &normalizer2, 1, PRECISION_ERR, "normalizer");

    free((void *)prob_new);
}

void test_joint_probs(void (*new_f)(Matrix *, Matrix *, Matrix *), Matrix *X, Matrix *P, Matrix *D){
    printf("Testing joint_probabilities:\n");
    Matrix D_new, X_new, P_new;
    copy_matrix(D, &D_new);
    copy_matrix(X, &X_new);
    copy_matrix(P, &P_new);

    joint_probs_baseline(X, P, D);
    new_f(&X_new, &P_new, &D_new);

    compareData(X->data, X_new.data, X->ncols*X->nrows, PRECISION_ERR, "X");
    compareData(D->data, D_new.data, D->ncols*D->nrows, PRECISION_ERR, "D");
    compareData(P->data, P_new.data, P->ncols*P->nrows, PRECISION_ERR, "P");
    
    free((void *)D_new.data);
    free((void *)X_new.data);
    free((void *)P_new.data);
}

void test_calc_affinities(void (*new_f)(Matrix *, Matrix *, Matrix *, Matrix *),
                            Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D){
    printf("Testing calc_affinities:\n");
    Matrix Yn, Qn, Q_num, Dn;
    copy_matrix(Y, &Yn);
    copy_matrix(Q, &Qn);
    copy_matrix(Q_numerators, &Q_num);
    copy_matrix(D, &Dn);

    calc_affinities(Y, Q, Q_numerators, D);
    new_f(&Yn, &Qn, &Q_num, &Dn);

    compareData(Y->data, Yn.data, Y->ncols*Y->nrows, PRECISION_ERR, "Y");
    compareData(Q->data, Qn.data, Q->ncols*Q->nrows, PRECISION_ERR, "Q");
    compareData(Q_numerators->data, Q_num.data, Q_numerators->ncols*Q_numerators->nrows, PRECISION_ERR, "Q_numerators");
    compareData(D->data, Dn.data, D->ncols*D->nrows, PRECISION_ERR, "D");
    
    free((void *)Yn.data);
    free((void *)Qn.data);
    free((void *)Q_num.data);
    free((void *)Dn.data);
}

void test_calc_cost(double (*new_f)(Matrix *, Matrix *), Matrix *P, Matrix *Q){
    printf("Testing calc_cost:\n");
    double res1, res2;
    
    res1 = calc_cost(P, Q);
    res2 = new_f(P, Q);

    compareData(&res1, &res2, 1, PRECISION_ERR, "cost");
}

static inline void copy_tsne_var(tsne_var_t *orig, tsne_var_t *copy){
    copy_matrix(&orig->D, &copy->D);
    copy_matrix(&orig->gains, &copy->gains);
    copy_matrix(&orig->grad_Y, &copy->grad_Y);
    copy_matrix(&orig->P, &copy->P);
    copy_matrix(&orig->Q, &copy->Q);
    copy_matrix(&orig->Q_numerators, &copy->Q_numerators);
    copy_matrix(&orig->tmp, &copy->tmp);
    copy_matrix(&orig->Y_delta, &copy->Y_delta);
}

static inline void free_tsne_var(tsne_var_t *var){
    free((void *)var->D.data);
    free((void *)var->gains.data);
    free((void *)var->grad_Y.data);
    free((void *)var->P.data);
    free((void *)var->Q.data);
    free((void *)var->Q_numerators.data);
    free((void *)var->tmp.data);
    free((void *)var->Y_delta.data);
}

static inline void compare_tsne_var(tsne_var_t *orig, tsne_var_t *alt){
    compareData(orig->D.data, alt->D.data, orig->D.ncols*orig->D.nrows, PRECISION_ERR, "D");
    compareData(orig->gains.data, alt->gains.data, orig->gains.ncols*orig->gains.nrows, PRECISION_ERR, "gains");
    compareData(orig->grad_Y.data, alt->grad_Y.data, orig->grad_Y.ncols*orig->grad_Y.nrows, PRECISION_ERR, "grad_Y");
    compareData(orig->P.data, alt->P.data, orig->P.ncols*orig->P.nrows, PRECISION_ERR, "P");
    compareData(orig->Q.data, alt->Q.data, orig->Q.ncols*orig->Q.nrows, PRECISION_ERR, "Q");
    compareData(orig->Q_numerators.data, alt->Q_numerators.data, orig->Q_numerators.ncols*orig->Q_numerators.nrows, PRECISION_ERR, "Q_numerator");
    compareData(orig->tmp.data, alt->tmp.data, orig->tmp.ncols*orig->tmp.nrows, PRECISION_ERR, "tmp");
    compareData(orig->Y_delta.data, alt->Y_delta.data, orig->Y_delta.ncols*orig->Y_delta.nrows, PRECISION_ERR, "Y_delta");
    }

void test_grad_desc(void (*new_f)(Matrix *, tsne_var_t *, int, int, double),
                        Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum) {
    printf("Testing gradient_descent:\n");
    Matrix Yn;
    tsne_var_t varn;
    copy_matrix(Y, &Yn);
    copy_tsne_var(var, &varn);

    grad_desc_baseline(Y, var, n, n_dim, momentum);
    new_f(&Yn, &varn, n, n_dim, momentum);

    compareData(Y->data, Yn.data, Y->ncols*Y->nrows, PRECISION_ERR, "Y");
    compare_tsne_var(var, &varn);

    free_tsne_var(&varn);
    free((void *)Yn.data);
}

void test_tsne(void (*new_f)(Matrix *, Matrix *, tsne_var_t *, int), Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim){
    printf("Testing tsne:\n");
    Matrix Yn, Xn;
    tsne_var_t varn;
    copy_matrix(Y, &Yn);
    copy_matrix(X, &Xn);
    copy_tsne_var(var, &varn);

    tsne_baseline(X, Y, var, n_dim);
    new_f(&Xn, &Yn, &varn, n_dim);

    compareData(X->data, Xn.data, X->ncols*X->nrows, PRECISION_ERR, "X");
    compareData(Y->data, Yn.data, Y->ncols*Y->nrows, PRECISION_ERR, "Y");
    compare_tsne_var(var, &varn);

    free_tsne_var(&varn);
    free((void *)Xn.data);
    free((void *)Yn.data);
}