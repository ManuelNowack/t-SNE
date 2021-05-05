#include "base.h"

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
void calc_squared_euclidean_distances(Matrix *X, Matrix *D) {

  int n = X->nrows;
  int m = X->ncols;

  // calculate non-diagonal entries
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Euclidean distance
      double sum = 0;
      for (int k = 0; k < m; k++) {
        double dist = X->data[i * m + k] - X->data[j * m + k];
        sum += dist * dist;
      }
      D->data[i * n + j] = sum;
      D->data[j * n + i] = sum;
    }
  }

  // set diagonal entries
  for (int i = 0; i < n; i++) {
    D->data[i * n + i] = 0.0;
  }
}

/*
* Calculate the log perplexity H for sample k for the given distances to the
* other element and precision. Precision is the inverse of two times the
* variance. Unnormalized probabilities, log perplexity, and normalizer of the
* probabilities are returned by reference.
*/
void calc_log_perplexity(double* distances, double* probabilities, int n, int k,
                         double precision, double* log_perplexity,
                         double* normlizer) {

  // calculate (unnormalised) conditional probabilities
  for (int i = 0; i < n; i++) {
    if (i == k) {
      probabilities[i] = 0;
    } else {
      probabilities[i] = exp(-precision * distances[i]);
    }
  }

  // normalisation
  double Z = 0;
  for (int i = 0; i < n; i++) {
    Z += probabilities[i];
  }

  // calculate log perplexity
  double H = 0;
  for (int i = 0; i < n; i++) {
    H += probabilities[i] * distances[i];
  }
  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}

/*
* calculates mean standard-deviation, used for debugging only
*/
double calculate_mean_stddev(double *precisions, int n){
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += sqrt(1 / (2 * precisions[i]));
  }
  return sum/n;
}

/*
* Calculate joint probabilities for high level points X with desired
* perplexity. Joint probabilities are stored in P, whose data field is
* expected to be suitably initialised with sufficient size. Actual perplexity
* may deviate at most tol from the desired log perplexity.
*/
void joint_probs_baseline(Matrix *X, Matrix *P, Matrix *D) {

  int n = X->nrows;

  calc_squared_euclidean_distances(X, D);

  double target_log_perplexity = log(kPerplexityTarget);

  // initialise precisions to 1
  double precisions[n];
  for (int i = 0; i < n; i++) {
    precisions[i] = 1;
  }

  // loop over all datapoints to determine precision and corresponding
  // probabilities
  for (int i = 0; i < n; i++) {
    double precision_min = 0.0;
    double precision_max = HUGE_VAL;
    double* distances = &D->data[i * n];
    double* probabilities = &P->data[i * n];

    // bisection method for a fixed number of iterations
    double actual_log_perplexity, normalizer, diff;
    for (int iter=0; iter<kJointProbsMaxIter; iter++) {

      calc_log_perplexity(distances, probabilities, n, i, precisions[i],
                          &actual_log_perplexity, &normalizer);
      diff = actual_log_perplexity - target_log_perplexity;

      if (diff > 0) {
        // precision should be increased
        precision_min = precisions[i];
        if (precision_max == HUGE_VAL) {
          precisions[i] *= 2;
        } else {
          precisions[i] = 0.5 * (precisions[i] + precision_max);
        }
      } else {
        // precision should be decreased
        precision_max = precisions[i];
        if (precision_min == 0.0) {
          precisions[i] /= 2;
        } else {
          precisions[i] = 0.5 * (precisions[i] + precision_min);
        }
      }
    }

    // normalize probabilities
    for (int i = 0; i < n; i++) {
      probabilities[i] = probabilities[i] / normalizer;
    }
  }

  DEBUG("Mean standard deviation: " << calculate_mean_stddev(precisions, n));

  // convert conditional probabilties to joint probabilities
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double a = P->data[i * n + j];
      double b = P->data[j * n + i];
      double prob = (a + b) / (2 * n);

      // early exaggeration
      prob *= 4;

      // ensure minimal probability
      if (prob < 1e-12) prob = 1e-12;

      P->data[i * n + j] = prob;
      P->data[j * n + i] = prob;
    }
  }
}

/*
* Calculate affinities as in Equation (4).
* The numerators of Equation (4) are stored in Q_numberators.
* All matrices are expected to be initialised with sufficient memory.
*/
void calc_affinities(Matrix *Y, Matrix *Q, Matrix *Q_numerators, Matrix *D) {

  int n = Y->nrows;

  // calculate squared Euclidean distances
  calc_squared_euclidean_distances(Y, D);

  // unnormalised perplexities
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = 1 / (1 + D->data[i * n + j]);
      Q_numerators->data[i * n + j] = value;
      Q_numerators->data[j * n + i] = value;
      sum += value;
    }
  }

  // set diagonal elements
  for (int i = 0; i < n; i++){
    Q->data[i * n + i] = 0;
  }

  // normalise
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = Q_numerators->data[i * n + j];
      value = 0.5 / sum * value;  // multiplication by 0.5, as sum only sum of
                                  // upper triangle elements

      // ensure minimum probability
      if (value < 1e-12) value = 1e-12;

      Q->data[i * n + j] = value;
      Q->data[j * n + i] = value;
    }
  }
}

/*
* Calculate cost (KL-divergence) between P and Q, i.e. KL(P||Q)
*/
double calc_cost(Matrix *P, Matrix *Q) {

  int n = P->nrows;

  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      sum += P->data[i * n + j] * log(P->data[i * n + j] / Q->data[i * n + j]);
    }
  }

  // double sum, as only upper triangular elements are summed due to symmetry of
  // P and Q
  return 2 * sum;
}

void grad_desc_baseline(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                        double momentum) {
  // calculate low-dimensional affinities
  calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);

  // calculate gradient with respect to embeddings Y
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double value = (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
                     var->Q_numerators.data[i * n + j];
      var->tmp.data[i * n + j] = value;
      var->tmp.data[j * n + i] = value;
    }
    var->tmp.data[i * n + i] = 0.0;
  }
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n_dim; k++) {
      double value = 0;
      for (int j = 0; j < n; j++) {
        value += var->tmp.data[i * n + j] *
                 (Y->data[i * n_dim + k] - Y->data[j * n_dim + k]);
      }
      value *= 4;
      var->grad_Y.data[i * n_dim + k] = value;
    }
  }

  // calculate gains, according to adaptive heuristic of Python implementation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      bool positive_grad = (var->grad_Y.data[i * n_dim + j] > 0);
      bool positive_delta = (var->Y_delta.data[i * n_dim + j] > 0);
      double value = var->gains.data[i * n_dim + j];
      if ((positive_grad && positive_delta) ||
          (!positive_grad && !positive_delta)) {
        value *= 0.8;
      } else {
        value += 0.2;
      }
      if (value < kMinGain) value = kMinGain;
      var->gains.data[i * n_dim + j] = value;
    }
  }

  // update step
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      double value =
          momentum * var->Y_delta.data[i * n_dim + j] -
          kEta * var->gains.data[i * n_dim + j] * var->grad_Y.data[i * n_dim + j];
      var->Y_delta.data[i * n_dim + j] = value;
      Y->data[i * n_dim + j] += value;
    }
  }

  // center each dimension at 0
  double means[n_dim];
  for (int j = 0; j < n_dim; j++) {
    means[j] = 0;
  }
  // accumulate
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      means[j] += Y->data[i * n_dim + j];
    }
  }
  // take mean
  for (int j = 0; j < n_dim; j++) {
    means[j] /= n;
  }
  // center
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      Y->data[i * n_dim + j] -= means[j];
    }
  }
}
/*
* Runs t-SNE on the matrix A, reducing its dimensionality to n_dim
* dimensions. Y is expected to contain the initial low dimensional
* embeddings. The values of Y are overwritten and at the end contain the
* calculated embeddings.
*/
void tsne_baseline(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim) {

  int n = X->nrows;

  // compute high level joint probabilities
  joint_probs_baseline(X, &var->P, &var->D);

  // determine embeddings
  // initialisations
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_dim; j++) {
      var->Y_delta.data[i * n_dim + j] = 0;
      var->gains.data[i * n_dim + j] = 1;
    }
  }

  double momentum = kInitialMomentum;
  for (int iter = 0; iter < kGradDescMaxIter; iter++) {
    // early exaggeration only for first 100 iterations
    if (iter == 100) {
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double value = var->P.data[i * n + j] / 4;
          var->P.data[i * n + j] = value;
          var->P.data[j * n + i] = value;
        }
      }
    }

    // reduce momentum at iteration 20
    if (iter == 20) momentum = kFinalMomentum;

    grad_desc_baseline(Y, var, n, n_dim, momentum);
  }
}
