/**
 * Holds the different gradient-descent functions
*/

#include <tsne/matrix.h>

void grad_desc_b(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum);

void grad_desc_ndim_unroll(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum);

void grad_desc_mean_unroll(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum);

void grad_desc_tmp_opt(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum);

void grad_desc_loop_merge(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum);

void grad_desc_accumulators(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum);
void grad_desc_accumulators2(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum);
void grad_desc_vectorized(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum);