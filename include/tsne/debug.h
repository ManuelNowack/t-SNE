#ifndef TSNE_DEBUG_H_
#define TSNE_DEBUG_H_

#include <iostream>

#ifdef DEBUG_BUILD
#define DEBUG(x)                 \
  do {                           \
    std::cerr << x << std::endl; \
  } while (0)
#else
#define DEBUG(x) \
  do {           \
  } while (0)
#endif

#endif  // TSNE_DEBUG_H_
