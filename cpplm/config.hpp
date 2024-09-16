#ifndef CPPLM_CONFIG_HPP
#define CPPLM_CONFIG_HPP
struct ModelConfig {
  int dim = 32;
  int n_heads = 4;
  int context_len = 128;
  int n_layers = 12;
  double dropout = 0.0;
  int vocab_size;
};
#endif