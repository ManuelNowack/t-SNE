#include <tsne/func_registry.h>
#include <tsne/matrix.h>

// Put all function declarations here.
tsne_func_t tsne_baseline;

void register_functions() {
  auto& func_registry = FuncResitry<tsne_func_t>::get_instance();

  // Put all functions to test here.
  func_registry.add_function(&tsne_baseline, "tsne_baseline");
}
