#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"



void calc_squared_euclidean_distances(Matrix X, Matrix D) {

  /*
   * Calculate pairwise squared Euclidean distances.
   * Squared Euclidean distances are stored in D, whose data field is expected to be suitably initialised with sufficient size.
   */

  int n = X.nrows;
  int m = X.ncols;

  // calculate non-diagonal entries
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      // Euclidean distance
      double sum = 0;
      for (int k=0; k<m; k++) {
        double dist = X.data[i*m + k] - X.data[j*m + k];
        sum += dist*dist;
      }
      D.data[i*n + j] = sum;
      D.data[j*n + i] = sum;
    }
  }

  // set diagonal entries
  for (int i=0; i<n; i++) {
    D.data[i*n + i] = 0.0;
  }
}

void calc_log_perplexity(double *distances, double *probabilities, int n, int k, double precision, double *log_perplexity, double *normlizer) {

  /*
   * Calculate the log perplexity H for sample k for the given distances to the other element and precision.
   * Precision is the inverse of two times the variance.
   * Unnormalized probabilities, log perplexity, and normalizer of the probabilities are returned by reference.
   */

  // calculate (unnormalised) conditional probabilities
  for (int i=0; i<n; i++) {
    if (i == k) {
      probabilities[i] = 0;
    } else {
      probabilities[i] = exp(-precision*distances[i]);
    }
  }

  // normalisation
  double Z = 0;
  for (int i=0; i<n; i++) {
    Z += probabilities[i];
  }

  // calculate log perplexity
  double H = 0;
  for (int i=0; i<n; i++) {
    H += probabilities[i]*distances[i];
  }
  H = precision*H/Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}

void calc_joint_probabilities(Matrix X, Matrix P, double perplexity, double tol) {

  /*
   * Calculate joint probabilities for high level points X with desired perplexity.
   * Joint probabilities are stored in P, whose data field is expected to be suitably initialised with sufficient size.
   * Actual perplexity may deviate at most tol from the desired log perplexity.
   */

  int n = X.nrows;

  Matrix D = create_matrix(n, n);
  calc_squared_euclidean_distances(X, D);

  double target_log_perplexity = log(perplexity);

  // initialise precisions to 1
  double precisions[n];
  for (int i=0; i<n; i++) {
    precisions[i] = 1;
  }

  // loop over all datapoints to determine precision and corresponding probabilities
  for (int i=0; i<n; i++) {

    if (i % 500 == 0) {
      printf("Computing probabilities for point %d of %d ...\n", i, n);
    }

    double precision_min = 0.0;
    double precision_max = HUGE_VAL;
    double *distances = &D.data[i*n];
    double *probabilities = &P.data[i*n];
    double actual_log_perplexity, normalizer;
    calc_log_perplexity(distances, probabilities, n, i, precisions[i], &actual_log_perplexity, &normalizer);

    // bisection method until suitable precision is found or maximum number of tries has been reached
    int tries = 0;
    double diff = actual_log_perplexity - target_log_perplexity;

    while (fabs(diff) > tol && tries < 50) {

      if (diff > 0) {
        // precision should be increased
        precision_min = precisions[i];
        if (precision_max == HUGE_VAL) {
          precisions[i] *= 2;
        } else {
          precisions[i] = 0.5*precisions[i] + 0.5*precision_max;
        }
      } else {
        // precision should be decreased
        precision_max = precisions[i];
        if (precision_min == 0.0) {
          precisions[i] /= 2;
        } else {
          precisions[i] = 0.5*(precisions[i] + precision_min);
        }
      }

      // calculate new log perplexity
      tries++;
      if (tries < 50) {
        calc_log_perplexity(distances, probabilities, n, i, precisions[i], &actual_log_perplexity, &normalizer);
        diff = actual_log_perplexity - target_log_perplexity;
      }
    }

    // normalize probabilities
    for (int i=0; i<n; i++) {
      probabilities[i] = probabilities[i]/normalizer;
    }
  }

  // calculate mean stdandard deviation
  double sum = 0;
  for (int i=0; i<n; i++) {
    sum += sqrt(1/(2*precisions[i]));
  }
  double mean_stddev = sum/n;
  printf("Mean standard deviation: %.3e\n", mean_stddev);

  // convert conditional probabilties to joint probabilities
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      double a = P.data[i*n + j];
      double b = P.data[j*n + i];
      double prob = (a + b)/(2*n);

      // early exaggeration
      prob *= 4;

      // ensure minimal probability
      if (prob < 1e-12) prob = 1e-12;

      P.data[i*n + j] = prob;
      P.data[j*n + i] = prob;
    }
  }
}

void calc_affinities(Matrix Y, Matrix Q, Matrix Q_numerators) {

  /*
   * Calculate affinities as in Equation (4).
   * The numerators of Equation (4) are stored in Q_numberators.
   * All matrices are expected to be initialised with sufficient memory.
   */

  int n = Y.nrows;

  // calculate squared Euclidean distances
  Matrix D = create_matrix(n, n);
  calc_squared_euclidean_distances(Y, D);

  // unnormalised perplexities
  double sum = 0;
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      double value = 1/(1 + D.data[i*n + j]);
      Q_numerators.data[i*n + j] = value;
      Q_numerators.data[j*n + i] = value;
      sum += value;
    }
  }

  // set diagonal elements
  for (int i=0; i<n; i++) Q.data[i*n + i] = 0;

  // normalise
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      double value = Q_numerators.data[i*n + j];
      value = 0.5/sum*value;  // multiplication by 0.5, as sum only sum of upper triangle elements
      
      // ensure minimum probability
      if (value < 1e-12) value = 1e-12;

      Q.data[i*n + j] = value;
      Q.data[j*n + i] = value;
    }
  }
}

double calc_cost(Matrix P, Matrix Q) {

  /*
   * Calculate cost (KL-divergence) between P and Q, i.e. KL(P||Q)
   */

  int n = P.nrows;

  double sum = 0;
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      sum += P.data[i*n + j]*log(P.data[i*n + j]/Q.data[i*n + j]);
    }
  }

  // double sum, as only upper triangular elements are summed due to symmetry of P and Q
  return 2*sum;
}

Matrix tsne(Matrix X, int n_dim, double perplexity, Matrix Y) {

  /*
   * Runs t-SNE on the matrix A, reducing its dimensionality to n_dim dimensions.
   * Y is expected to contain the initial low dimensional embeddings.
   * The values of Y are overwritten and at the end contain the calculated embeddings.
   */

  // hyperparameters
  int max_iter = 1000;
  double initial_momentum = 0.5;
  double final_momentum = 0.8;
  double eta = 500;
  double min_gain = 0.01;
  double rand_max_inv = 1.0/RAND_MAX;

  int n = X.nrows;

  // compute high level joint probabilities
  Matrix P = create_matrix(n, n);
  calc_joint_probabilities(X, P, perplexity, 1e-5);

  // determine embeddings
  Matrix Q = create_matrix(n, n);
  Matrix Q_numerators = create_matrix(n, n);
  Matrix grad_Y = create_matrix(n, n_dim);
  Matrix Y_delta = create_matrix(n, n_dim);
  Matrix tmp = create_matrix(n, n);
  Matrix gains = create_matrix(n, n_dim);

  // initialisations
  for (int i=0; i<n; i++) {
    for (int j=0; j<n_dim; j++) {
      Y_delta.data[i*n_dim + j] = 0;
      gains.data[i*n_dim + j] = 1;
    }
  }

  double momentum = initial_momentum;
  for (int iter=0; iter<max_iter; iter++) {

    // early exaggeration only for first 100 iterations
    if (iter == 100) {
      for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
          double value = P.data[i*n + j]/4;
          P.data[i*n + j] = value;
          P.data[j*n + i] = value;
        }
      }
    }

    // reduce momentum at iteration 20
    if (iter == 20) momentum = final_momentum;

    // calculate low-dimensional affinities
    calc_affinities(Y, Q, Q_numerators);

    // calculate gradient with respect to embeddings Y
    for (int i=0; i<n; i++) {
      for (int j=i+1; j<n; j++) {
        double value = (P.data[i*n + j] - Q.data[i*n + j])*Q_numerators.data[i*n + j];
        tmp.data[i*n + j] = value;
        tmp.data[j*n + i] = value;
      }
      tmp.data[i*n + i] = 0.0;
    }
    for (int i=0; i<n; i++) {
      for (int k=0; k<n_dim; k++) {
        double value = 0;
        for (int j=0; j<n; j++) {
          value += tmp.data[i*n + j]*(Y.data[i*n_dim + k] - Y.data[j*n_dim + k]);
        }
        value *= 4;
        grad_Y.data[i*n_dim + k] = value;
      }
    }

    // calculate gains, according to adaptive heuristic of Python implementation
    for (int i=0; i<n; i++) {
      for (int j=0; j<n_dim; j++) {
        bool positive_grad = (grad_Y.data[i*n_dim + j] > 0);
        bool positive_delta = (Y_delta.data[i*n_dim + j] > 0);
        double value = gains.data[i*n_dim + j];
        if ((positive_grad && positive_delta) || (!positive_grad && !positive_delta)) {
          value *= 0.8;
        } else {
          value += 0.2;
        }
        if (value < min_gain) value = min_gain;
        gains.data[i*n_dim + j] = value;
      }
    }

    // update step
    for (int i=0; i<n; i++) {
      for (int j=0; j<n_dim; j++) {
        double value = momentum*Y_delta.data[i*n_dim + j] - eta*gains.data[i*n_dim + j]*grad_Y.data[i*n_dim + j];
        Y_delta.data[i*n_dim + j] = value;
        Y.data[i*n_dim + j] += value;
      }
    }

    // center each dimension at 0
    double means[n_dim];
    for (int j=0; j<n_dim; j++) {
      means[j] = 0;
    }
    // accumulate
    for (int i=0; i<n; i++) {
      for (int j=0; j<n_dim; j++) {
        means[j] += Y.data[i*n_dim + j];
      }
    }
    // take mean
    for (int j=0; j<n_dim; j++) {
      means[j] /= n;
    }
    // center
    for (int i=0; i<n; i++) {
      for (int j=0; j<n_dim; j++) {
        Y.data[i*n_dim + j] -= means[j];
      }
    }

    // log cost function
    if ((iter + 1) % 10 == 0) {
      double cost = calc_cost(P, Q);
      printf("Iteration %d | Cost %.3e\n", iter+1, cost);
    }

  }
  
  return Y;
}

int main() {
  printf("Running example on 2,500 MNIST digits...\n");
  Matrix X = load_matrix("mnist2500_X_pca.txt");
  Matrix Y = load_matrix("mnist2500_Y_init.txt");
  tsne(X, 2, 20, Y);
  store_matrix("mnist2500_Y.txt", Y);
  return 0;
}