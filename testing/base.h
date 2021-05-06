#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tsne/debug.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

/**
 * Holds the ground truth function which further implementations of the respective
 * functions will be compared to.
 * 
*/



/*
* Calculate pairwise squared Euclidean distances.
* Squared Euclidean distances are stored in D, whose data field is expected
* to be suitably initialised with sufficient size.
*/
void euclidean_dist_baseline(Matrix *X, Matrix *D);

/*
* Calculate the log perplexity H for sample k for the given distances to the
* other element and precision. Precision is the inverse of two times the
* variance. Unnormalized probabilities, log perplexity, and normalizer of the
* probabilities are returned by reference.
*/
void calc_log_perplexity(double* distances, double* probabilities, int n, int k,
                         double precision, double* log_perplexity,
                         double* normlizer);

/*
* calculates mean standard-deviation, used for debugging only
*/
double calculate_mean_stddev(double *precisions, int n);

/*
* Calculate joint probabilities for high level points X with desired
* perplexity. Joint probabilities are stored in P, whose data field is
* expected to be suitably initialised with sufficient size. Actual perplexity
* may deviate at most tol from the desired log perplexity.
*/
void joint_probs_baseline(Matrix *X, Matrix *P, Matrix *D);

/*
* Calculate affinities as in Equation (4).
* The numerators of Equation (4) are stored in Q_numberators.
* All matrices are expected to be initialised with sufficient memory.
*/
void calc_affinities(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D);

/*
* Calculate cost (KL-divergence) between P and Q, i.e. KL(P||Q)
*/
double calc_cost(Matrix *P, Matrix *Q);

void grad_desc_baseline(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum);
/*
* Runs t-SNE on the matrix A, reducing its dimensionality to n_dim
* dimensions. Y is expected to contain the initial low dimensional
* embeddings. The values of Y are overwritten and at the end contain the
* calculated embeddings.
*/
void tsne_baseline(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim);