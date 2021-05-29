#ifndef TSNE_BENCHMARK_H_
#define TSNE_BENCHMARK_H_

#include <tsne/func_registry.h>
#include <tsne/matrix.h>

#define CYCLES_REQUIRED (1 * 1e8)
#define REP 5  // 50

void create_tsne_variables(tsne_var_t &var, int n, int n_dim);
void destroy_tsne_variables(tsne_var_t &var);
double perf_test_tsne(tsne_func_t *f, Matrix &X, Matrix &Y);
double perf_test_joint_probs(joint_probs_func_t *f, Matrix &X);
double perf_test_grad_desc(grad_desc_func_t *f, Matrix &X, Matrix &Y);
double perf_test_log_perplexity(log_perplexity_func_t *f, Matrix &X);
double perf_test_euclidean_dist(euclidean_dist_func_t *f, Matrix &X);

#endif  // TSNE_BENCHMARK_H_
