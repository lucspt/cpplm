#ifndef CPPLM_CONFIG_HPP
#define CPPLM_CONFIG_HPP

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
  int dim{32};
  int n_heads{4};
  int context_len{128};
  int n_layers{12};
  double dropout{0.0};
  double rmsnorm_eps{1e-6};
  int vocab_size{};

  ModelConfig() = default;
  ModelConfig(const int& vocab_size) : vocab_size{vocab_size} {};
};
#endif
