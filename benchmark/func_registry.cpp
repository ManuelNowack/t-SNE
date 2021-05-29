#include <tsne/func_registry.h>
#include <tsne/matrix.h>

void register_functions() {
  auto &tsne_func_registry = FuncRegistry<tsne_func_t>::get_instance();
  auto &joint_probs_func_registry =
      FuncRegistry<joint_probs_func_t>::get_instance();
  auto &grad_desc_func_registry =
      FuncRegistry<grad_desc_func_t>::get_instance();
  auto &log_perplexity_func_registry =
      FuncRegistry<log_perplexity_func_t>::get_instance();
  auto &euclidean_dist_func_registry =
      FuncRegistry<euclidean_dist_func_t>::get_instance();

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
      .add_function(&log_perplexity_unroll2, "log_perplexity_unroll2")
      .add_function(&log_perplexity_unroll4, "log_perplexity_unroll4")
      .add_function(&log_perplexity_unroll8, "log_perplexity_unroll8")
      .add_function(&log_perplexity_avx, "log_perplexity_avx")
      .add_function(&log_perplexity_avx_acc4, "log_perplexity_avx_acc4");
      
  // Put all euclidean_dist functions to test here.
  euclidean_dist_func_registry.add_function(&euclidean_dist_baseline,
                                            "euclidean_dist_baseline");
}
