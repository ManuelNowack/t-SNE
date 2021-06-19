#ifndef TSNE_FUNC_REGISTRY_H_
#define TSNE_FUNC_REGISTRY_H_

#include <tsne/matrix.h>

#include <string>
#include <vector>

typedef void tsne_func_t(Matrix *X, Matrix *Y, tsne_var_t *var, int n_dim);
typedef void joint_probs_func_t(Matrix *X, Matrix *P, Matrix *D);
typedef void grad_desc_func_t(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                              double momentum);
typedef void euclidean_dist_func_t(Matrix *X, Matrix *D);
typedef void log_perplexity_func_t(double *distances, double *probabilities,
                                   int n, int k, double precision,
                                   double *log_perplexity, double *normlizer);

// Put all log_perp_actual fuction declarations here.
log_perplexity_func_t log_perplexity_baseline, log_perplexity_unroll2,
    log_perplexity_unroll4, log_perplexity_unroll8, log_perplexity_avx,
    log_perplexity_avx_acc4, log_perplexity_avx_fma_acc4;

// Put all tsne function declarations here.
tsne_func_t tsne_baseline, tsne_no_vars, tsne_scalar, tsne_vec, tsne_vec2,
    tsne_vec3;

// Put all joint_probs function declarations here.
joint_probs_func_t joint_probs_baseline, joint_probs_unroll8,
    joint_probs_avx_fma_acc4;

// Put all grad_desc function declarations here.
grad_desc_func_t grad_desc_baseline, grad_desc_ndim_unroll,
    grad_desc_mean_unroll, grad_desc_tmp_opt, grad_desc_loop_merge,
    grad_desc_accumulators2, grad_desc_accumulators, grad_desc_vec_bottom,
    grad_desc_vectorized, _grad_desc_vec, _grad_desc_vec2, _grad_desc_vec3,
    grad_desc_no_vars_baseline, grad_desc_no_vars_tmp,
    grad_desc_no_vars_D, grad_desc_no_vars_Q, grad_desc_no_vars_Q_numerators,
    grad_desc_no_vars_scalar, grad_desc_no_vars_no_if,
    grad_desc_no_vars_grad, grad_desc_no_vars_means,
    grad_desc_no_vars_unroll2, grad_desc_no_vars_unroll4,
    grad_desc_no_vars_unroll6, grad_desc_no_vars_unroll8,
    grad_desc_no_vars_fetch, grad_desc_no_vars_no_l,
    grad_desc_no_vars_unroll, grad_desc_no_vars_vector,
    grad_desc_no_vars_vector_all, grad_desc_no_vars_vector_unroll2;

// Put all euclidean_dist function declarations here.
euclidean_dist_func_t euclidean_dist_baseline, euclidean_dist_unroll2,
    euclidean_dist_unroll4, euclidean_dist_unroll8, euclidean_dist_block8,
    euclidean_dist_block8x8, euclidean_dist_alt_baseline,
    euclidean_dist_alt_unroll2, euclidean_dist_alt_unroll4,
    euclidean_dist_alt_unroll8, euclidean_dist_alt_unroll16,
    euclidean_dist_alt_block4x4, euclidean_dist_alt_vec,
    euclidean_dist_alt_vec_unroll2, euclidean_dist_alt_vec_unroll4,
    euclidean_dist_alt_vec_unroll8, euclidean_dist_alt_vec_unroll4x4,
    euclidean_dist_low_upper, euclidean_dist_low_unroll,
    euclidean_dist_low_block2, euclidean_dist_low_block4,
    euclidean_dist_low_block8, euclidean_dist_low_block16,
    euclidean_dist_low_block32, euclidean_dist_low_block64,
    euclidean_dist_low_block128, euclidean_dist_low_vec1,
    euclidean_dist_low_vec2, euclidean_dist_low_vec3, euclidean_dist_low_vec4,
    euclidean_dist_low_vec3_unroll2, euclidean_dist_low_vec3_unroll4,
    euclidean_dist_low_vec3_unroll8, euclidean_dist_low_vec3_unroll4x8,
    euclidean_dist_low_vec3_unroll4x8_stream,
    euclidean_dist_low_vec3_unroll8_stream;

template <class T>
class FuncRegistry {
 public:
  std::vector<T *> funcs;
  std::vector<std::string> func_names;
  int num_funcs = 0;

  // Constraints on the singleton.
  FuncRegistry(FuncRegistry const &) = delete;
  void operator=(FuncRegistry const &) = delete;

  // Get the single instance of the FunctionRegistry.
  static FuncRegistry &get_instance() {
    static FuncRegistry instance;
    return instance;
  }

  // Registers a user function to be tested by the driver program. Registers a
  // string description of the function as well.
  FuncRegistry &add_function(T *f, std::string name) {
    funcs.push_back(f);
    func_names.emplace_back(name);
    num_funcs++;
    return *this;
  }

 private:
  FuncRegistry(){};
};

// Called by the driver to register your functions.
void register_functions();

#endif  // TSNE_FUNC_REGISTRY_H_
