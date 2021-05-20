#include <immintrin.h>
#include <math.h>
#include <tsne/debug.h>
#include <tsne/hyperparams.h>
#include <tsne/matrix.h>
#include <xmmintrin.h>

void log_perplexity_unroll2(double *distances, double *probabilities, int n,
                            int k, double precision, double *log_perplexity,
                            double *normlizer) {
  // calculate unnormalised conditional probabilities and normalization.
  double Z = 0, H = 0;
  int i = 0;
  for (; i < n - 1; i += 2) {
    double d0 = distances[i];
    double d1 = distances[i + 1];

    double p0 = exp(-precision * d0);
    double p1 = exp(-precision * d1);

    Z += p0 + p1;
    H += p0 * d0 + p1 * d1;

    probabilities[i] = p0;
    probabilities[i + 1] = p1;
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

void log_perplexity_unroll8(double *distances, double *probabilities, int n,
                            int k, double precision, double *log_perplexity,
                            double *normlizer) {
  // calculate unnormalised conditional probabilities and normalization.
  double Z = 0, H = 0;
  int i = 0;
  for (; i < n - 7; i += 8) {
    double d0 = distances[i];
    double d1 = distances[i + 1];
    double d2 = distances[i + 2];
    double d3 = distances[i + 3];
    double d4 = distances[i + 4];
    double d5 = distances[i + 5];
    double d6 = distances[i + 6];
    double d7 = distances[i + 7];

    double p0 = exp(-precision * d0);
    double p1 = exp(-precision * d1);
    double p2 = exp(-precision * d2);
    double p3 = exp(-precision * d3);
    double p4 = exp(-precision * d4);
    double p5 = exp(-precision * d5);
    double p6 = exp(-precision * d6);
    double p7 = exp(-precision * d7);

    Z += p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
    H += p0 * d0 + p1 * d1 + p2 * d2 + p3 * d3 + p4 * d4 + p5 * d5 + p6 * d6 +
         p7 * d7;

    probabilities[i] = p0;
    probabilities[i + 1] = p1;
    probabilities[i + 2] = p2;
    probabilities[i + 3] = p3;
    probabilities[i + 4] = p4;
    probabilities[i + 5] = p5;
    probabilities[i + 6] = p6;
    probabilities[i + 7] = p7;
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

void log_perplexity_avx(double *distances, double *probabilities, int n, int k,
                        double precision, double *log_perplexity,
                        double *normlizer) {}
