#include <tsne/matrix.h>
#include <tsne/grad_descent.h>
#include "base.h"
#include "tests.h"


// Create intermediate t-SNE variables.
static inline void create_tsne_variables(tsne_var_t &var, int n, int n_dim) {
    var.P = create_matrix(n, n);
    var.Q = create_matrix(n, n);
    var.Q_numerators = create_matrix(n, n);
    var.grad_Y = create_matrix(n, n_dim);
    var.Y_delta = create_matrix(n, n_dim);
    var.tmp = create_matrix(n, n);
    var.gains = create_matrix(n, n_dim);
    var.D = create_matrix(n, n);
}

static inline void destroy_tsne_variables(tsne_var_t &var) {
    free(var.P.data);
    free(var.Q.data);
    free(var.Q_numerators.data);
    free(var.grad_Y.data);
    free(var.Y_delta.data);
    free(var.tmp.data);
    free(var.gains.data);
    free(var.D.data);
}

int main(int argc, char **argv) {
    Matrix X = load_matrix(argv[1]);

    Matrix Y = load_matrix(argv[2]);

    tsne_var_t var;
    create_tsne_variables(var, X.nrows, 2);
    joint_probs_baseline(&X, &var.P, &var.D);
    test_grad_desc(grad_desc_vec_bottom, &Y, &var, X.nrows, 2, 0.8);
}