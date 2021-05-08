#include <tsne/func_registry.h>
#include <tsne/matrix.h>

// Put all tsne function declarations here.
tsne_func_t tsne_baseline;

// Put all joint_probs function declarations here.
joint_probs_func_t joint_probs_baseline;

// Put all grad_desc function declarations here.
grad_desc_func_t grad_desc_baseline;

// Put all euclidean_dist function declarations here:
euclidean_dist_func_t euclidean_dist_baseline;
euclidean_dist_func_t euclidean_dist_unroll2;
euclidean_dist_func_t euclidean_dist_unroll4;
euclidean_dist_func_t euclidean_dist_unroll8;

void register_functions() {
  auto &tsne_func_registry = FuncRegistry<tsne_func_t>::get_instance();
  auto &joint_probs_func_registry =
      FuncRegistry<joint_probs_func_t>::get_instance();
  auto &grad_desc_func_registry = FuncRegistry<grad_desc_func_t>::get_instance();
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

  // Put all euclidean_dist functions to test here.
  euclidean_dist_func_registry.add_function(&euclidean_dist_baseline, "euclidean_dist_baseline");
  euclidean_dist_func_registry.add_function(&euclidean_dist_unroll2, "euclidean_dist_unroll2");
  euclidean_dist_func_registry.add_function(&euclidean_dist_unroll4, "euclidean_dist_unroll4");
  euclidean_dist_func_registry.add_function(&euclidean_dist_unroll8, "euclidean_dist_unroll8");
}
