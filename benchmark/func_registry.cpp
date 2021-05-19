#include <tsne/func_registry.h>
#include <tsne/matrix.h>

// Put all tsne function declarations here.
tsne_func_t tsne_baseline;

// Put all joint_probs function declarations here.
joint_probs_func_t joint_probs_baseline;

// Put all grad_desc function declarations here.
grad_desc_func_t grad_desc_baseline;

// Put all log_perp_actual fuction declarations here.
log_perplexity_func_t log_perplexity_baseline, log_perplexity_unroll4;

void register_functions() {
  auto &tsne_func_registry = FuncRegistry<tsne_func_t>::get_instance();
  auto &joint_probs_func_registry =
      FuncRegistry<joint_probs_func_t>::get_instance();
  auto &grad_desc_func_registry =
      FuncRegistry<grad_desc_func_t>::get_instance();
  auto &log_perplexity_func_registry =
      FuncRegistry<log_perplexity_func_t>::get_instance();

  // Put all tsne functions to test here.
  tsne_func_registry.add_function(&tsne_baseline, "tsne_baseline");

  // Put all (at least one!) joint_probs functions to test here.
  joint_probs_func_registry.add_function(&joint_probs_baseline,
                                         "joint_probs_baseline");

  // Put all grad_desc functions to test here.
  grad_desc_func_registry.add_function(&grad_desc_baseline,
                                       "grad_desc_baseline");

  // Put all log_perplexity functions to test here.
  log_perplexity_func_registry
      .add_function(&log_perplexity_baseline, "log_perplexity_baseline")
      .add_function(&log_perplexity_unroll4, "log_perplexity_unroll4");
}
