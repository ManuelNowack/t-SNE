#include <tsne/func_registry.h>
#include <tsne/matrix.h>

// Put all tsne function declarations here.
tsne_func_t tsne_baseline;

// Put all joint_probs function declarations here.
joint_probs_func_t joint_probs_baseline;

// Put all grad_desc function declarations here.
grad_desc_func_t grad_desc_baseline;

void register_functions() {
  auto &tsne_func_registry = FuncResitry<tsne_func_t>::get_instance();
  auto &joint_probs_func_registry =
      FuncResitry<joint_probs_func_t>::get_instance();
  auto &grad_desc_func_registry = FuncResitry<grad_desc_func_t>::get_instance();

  // Put all tsne functions to test here.
  tsne_func_registry.add_function(&tsne_baseline, "tsne_baseline");

  // Put all (at least one!) joint_probs functions to test here.
  joint_probs_func_registry.add_function(&joint_probs_baseline,
                                         "joint_probs_baseline");

  // Put all grad_desc functions to test here.
  grad_desc_func_registry.add_function(&grad_desc_baseline,
                                       "grad_desc_baseline");
}