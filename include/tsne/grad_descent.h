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

void grad_desc_ydata_opt(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum);