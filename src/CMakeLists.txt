add_library(tsne_library
    matrix.cpp
    baseline.cpp
    log_perplexity.cpp
    joint_probs.cpp
    affinities.cpp
    euclidean_dist.cpp
    euclidean_dist_low.cpp
    grad_desc.cpp
    tsne.cpp
)

# We need this directory, and users of our library will need it too
target_include_directories(tsne_library PUBLIC ../include)
