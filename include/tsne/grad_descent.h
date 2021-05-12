/**
 * Holds the different gradient-descent functions
*/

#include <tsne/matrix.h>

void grad_desc_baseline(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum);