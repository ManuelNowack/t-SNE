#ifndef TSNE_FUNC_REGISTRY_H_
#define TSNE_FUNC_REGISTRY_H_

#include <tsne/matrix.h>

#include <string>
#include <vector>

typedef void tsne_func_t(Matrix *X, Matrix *Y, tsne_var_t *var,
                         int n_dim);
typedef void joint_probs_func_t(Matrix *X, Matrix *P, Matrix *D);
typedef void grad_desc_func_t(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                              double momentum);

template <class T>
class FuncRegistry {
 public:
  std::vector<T*> funcs;
  std::vector<std::string> func_names;
  int num_funcs = 0;

  // Constraints on the singleton.
  FuncRegistry(FuncRegistry const&) = delete;
  void operator=(FuncRegistry const&) = delete;

  // Get the single instance of the FunctionRegistry.
  static FuncRegistry& get_instance() {
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
