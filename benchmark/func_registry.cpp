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
  joint_probs_func_registry
      .add_function(&joint_probs_baseline, "joint_probs_baseline")
      .add_function(&joint_probs_unroll8, "joint_probs_unroll8")
      .add_function(&joint_probs_avx_fma_acc4, "joint_probs_avx_fma_acc4");

  // Put all grad_desc functions to test here.
  grad_desc_func_registry.add_function(&grad_desc_baseline,"grad_desc_baseline")
  //.add_function(&grad_desc_ndim_unroll,"grad_desc_ndim_unroll")
  .add_function(&grad_desc_mean_unroll,"grad_desc_mean_unroll")
  .add_function(&grad_desc_tmp_opt,"grad_desc_tmp_opt")
  .add_function(&grad_desc_loop_merge,"grad_desc_loop_merge")
  .add_function(&grad_desc_accumulators,"grad_desc_accumulators")
  .add_function(&grad_desc_accumulators2,"grad_desc_accumulators2")
  //.add_function(&grad_desc_vec_bottom,"grad_desc_vec_bottom")
  .add_function(&grad_desc_vectorized,"grad_desc_vectorized");

  // Put all log_perplexity functions to test here.
  log_perplexity_func_registry
      .add_function(&log_perplexity_baseline, "log_perplexity_baseline")
      .add_function(&log_perplexity_unroll2, "log_perplexity_unroll2")
      .add_function(&log_perplexity_unroll4, "log_perplexity_unroll4")
      .add_function(&log_perplexity_unroll8, "log_perplexity_unroll8")
      .add_function(&log_perplexity_avx, "log_perplexity_avx")
      .add_function(&log_perplexity_avx_acc4, "log_perplexity_avx_acc4")
      .add_function(&log_perplexity_avx_fma_acc4,
                    "log_perplexity_avx_fma_acc4");

  // Put all euclidean_dist functions to test here.
  euclidean_dist_func_registry
      .add_function(&euclidean_dist_baseline, "euclidean_dist_baseline");
      /*
      .add_function(&euclidean_dist_unroll2, "euclidean_dist_unroll2")
      .add_function(&euclidean_dist_unroll4, "euclidean_dist_unroll4")
      .add_function(&euclidean_dist_unroll8, "euclidean_dist_unroll8")
      .add_function(&euclidean_dist_block8, "euclidean_dist_block8")
      .add_function(&euclidean_dist_block8x8, "euclidean_dist_block8x8")
      .add_function(&euclidean_dist_alt_baseline, "euclidean_dist_alt_baseline")
      .add_function(&euclidean_dist_alt_unroll2, "euclidean_dist_alt_unroll2")
      .add_function(&euclidean_dist_alt_unroll4, "euclidean_dist_alt_unroll4")
      .add_function(&euclidean_dist_alt_unroll8, "euclidean_dist_alt_unroll8")
      .add_function(&euclidean_dist_alt_unroll16, "euclidean_dist_alt_unroll16")
      .add_function(&euclidean_dist_alt_block4x4, "euclidean_dist_alt_block4x4")
      .add_function(&euclidean_dist_alt_vec, "euclidean_dist_alt_vec")
      .add_function(&euclidean_dist_alt_vec_unroll2,
                    "euclidean_dist_alt_vec_unroll2")
      .add_function(&euclidean_dist_alt_vec_unroll4,
                    "euclidean_dist_alt_vec_unroll4")
      .add_function(&euclidean_dist_alt_vec_unroll8,
                    "euclidean_dist_alt_vec_unroll8")
      .add_function(&euclidean_dist_alt_vec_unroll4x4,
                    "euclidean_dist_alt_vec_unroll4x4")
      .add_function(&euclidean_dist_low_upper, "euclidean_dist_low_upper")
      .add_function(&euclidean_dist_low_unroll, "euclidean_dist_low_unroll")
      .add_function(&euclidean_dist_low_block2, "euclidean_dist_low_block2")
      .add_function(&euclidean_dist_low_block4, "euclidean_dist_low_block4")
      .add_function(&euclidean_dist_low_block8, "euclidean_dist_low_block8")
      .add_function(&euclidean_dist_low_block16, "euclidean_dist_low_block16")
      .add_function(&euclidean_dist_low_block32, "euclidean_dist_low_block32")
      .add_function(&euclidean_dist_low_block64, "euclidean_dist_low_block64")
      .add_function(&euclidean_dist_low_block128, "euclidean_dist_low_block128")
      .add_function(&euclidean_dist_low_vec1, "euclidean_dist_low_vec1")
      .add_function(&euclidean_dist_low_vec2, "euclidean_dist_low_vec2")
      .add_function(&euclidean_dist_low_vec3, "euclidean_dist_low_vec3")
      .add_function(&euclidean_dist_low_vec4, "euclidean_dist_low_vec4")
      .add_function(&euclidean_dist_low_vec3_unroll2,
                    "euclidean_dist_low_vec3_unroll2")
      .add_function(&euclidean_dist_low_vec3_unroll4,
                    "euclidean_dist_low_vec3_unroll4")
      .add_function(&euclidean_dist_low_vec3_unroll8,
                    "euclidean_dist_low_vec3_unroll8")
      .add_function(&euclidean_dist_low_vec3_unroll4x8,
                    "euclidean_dist_low_vec3_unroll4x8")
      .add_function(&euclidean_dist_low_vec3_unroll4x8_stream,
                    "euclidean_dist_low_vec3_unroll4x8_stream")
      .add_function(&euclidean_dist_low_vec3_unroll8_stream,
                    "euclidean_dist_low_vec3_unroll8_stream");*/
}
