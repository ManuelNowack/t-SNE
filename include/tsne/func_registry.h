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
    log_perplexity_avx_acc4;

// Put all tsne function declarations here.
tsne_func_t tsne_baseline;

// Put all joint_probs function declarations here.
joint_probs_func_t joint_probs_baseline;

// Put all grad_desc function declarations here.
grad_desc_func_t grad_desc_baseline;

// Put all euclidean_dist function declarations here.
euclidean_dist_func_t euclidean_dist_baseline;

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
