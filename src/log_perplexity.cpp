#include <math.h>
#include <tsne/debug.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>

void log_perplexity_unroll4(double *distances, double *probabilities, int n,
                            int k, double precision, double *log_perplexity,
                            double *normlizer) {
  // calculate unnormalised conditional probabilities and normalization.
  double Z = 0, H = 0;
  int i = 0;
  for (; i < n - 3; i += 4) {
    double d0 = distances[i];
    double d1 = distances[i + 1];
    double d2 = distances[i + 2];
    double d3 = distances[i + 3];

    double p0 = exp(-precision * d0);
    double p1 = exp(-precision * d1);
    double p2 = exp(-precision * d2);
    double p3 = exp(-precision * d3);

    Z += p0 + p1 + p2 + p3;
    H += p0 * d0 + p1 * d1 + p2 * d2 + p3 * d3;

    probabilities[i] = p0;
    probabilities[i + 1] = p1;
    probabilities[i + 2] = p2;
    probabilities[i + 3] = p3;
  }

  for (; i < n; i++) {
    double di = distances[i];
    double pi = exp(-precision * di);
    Z += pi;
    H += pi * di;
    probabilities[i] = pi;
  }

  double pk = probabilities[k];
  Z -= pk;
  H -= pk * distances[k];
  probabilities[k] = 0;

  H = precision * H / Z + log(Z);

  *log_perplexity = H;
  *normlizer = Z;
}
