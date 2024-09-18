#ifndef CPPLM_CONFIG_HPP
#define CPPLM_CONFIG_HPP

constexpr int MODEL_CONFIG_DIM{32};
constexpr int MODEL_CONFIG_N_HEADS{4};
constexpr int MODEL_CONFIG_CONTEXT_LEN{128};
constexpr int MODEL_CONFIG_N_LAYERS{12};
constexpr double MODEL_CONFIG_DROPOUT{0.0};
constexpr double MODEL_CONFIG_RMSNORM_EPS{1e-6};

/**
 * Model configuration
 * 
 * @param dim Number of embedding dimensions
 * @param n_heads Number of attention heads
 * @param context_len The max length of a sequence (aka seqlen)
 * @param n_layers Number of transformer layers
 * @param dropout Dropout probability factor 
 * @param rmsnorm_eps RMSNorm epsilon value
 * @param vocab_size The vocabulary size (number of unique tokens the model can learn).
 */
struct ModelConfig {
  int dim{MODEL_CONFIG_DIM};
  int n_heads{MODEL_CONFIG_N_HEADS};
  int context_len{MODEL_CONFIG_CONTEXT_LEN};
  int n_layers{MODEL_CONFIG_N_LAYERS};
  double dropout{MODEL_CONFIG_DROPOUT};
  double rmsnorm_eps{MODEL_CONFIG_RMSNORM_EPS};
  int vocab_size{};

  ModelConfig() = default;
  ModelConfig(const int& vocab_size) : vocab_size{vocab_size} {};
};
#endif
