#ifndef TSNE_HYPERPARAMS_H_
#define TSNE_HYPERPARAMS_H_

constexpr int kGradDescMaxIter = 100;
constexpr int kJointProbsMaxIter = 50;
constexpr double kInitialMomentum = 0.5;
constexpr double kFinalMomentum = 0.8;
constexpr double kEta = 500;
constexpr double fourkEta = 4*kEta;
constexpr double kMinGain = 0.01;
constexpr double kPerplexityTarget = 20;
constexpr double kPerplexityTolerance = 1e-5;
constexpr double kMinimumProbability = 1e-12;

#endif  // TSNE_HYPERPARAMS_H_
