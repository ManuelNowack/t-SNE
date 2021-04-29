#ifndef TSNE_FUNC_REGISTRY_H_
#define TSNE_FUNC_REGISTRY_H_

#include <tsne/matrix.h>

#include <string>
#include <vector>

typedef void tsne_func_t(Matrix, int, double, Matrix);

class FuncResitry {
 public:
  std::vector<tsne_func_t*> funcs;
  std::vector<std::string> func_names;
  int num_funcs = 0;

  // Constraints on the singleton.
  FuncResitry(FuncResitry const&) = delete;
  void operator=(FuncResitry const&) = delete;

  // Get the single instance of the FunctionRegistry.
  static FuncResitry& get_instance() {
    static FuncResitry instance;
    return instance;
  }

  // Registers a user function to be tested by the driver program. Registers a
  // string description of the function as well.
  void add_function(tsne_func_t* f, std::string name) {
    funcs.push_back(f);
    func_names.emplace_back(name);
    num_funcs++;
  }

 private:
  FuncResitry(){};
};

// Put all function declarations here.
tsne_func_t tsne_baseline;

// Called by the driver to register your functions.
void register_functions() {
  FuncResitry& func_registry = FuncResitry::get_instance();

  // Put all functions to test here.
  func_registry.add_function(&tsne_baseline, "tsne_baseline");
}

#endif  // TSNE_FUNC_REGISTRY_H_