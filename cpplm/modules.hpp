#ifndef CPPLM_MODULES_HPP
#define CPPLM_MODULES_HPP
#include <torch/torch.h>
#include <config.hpp>
#include <string>

/**
 * Implements root mean square normalization. 
 * 
 * See here: https://arxiv.org/abs/1910.07467
 */
class RMSNormImpl : public torch::nn::Module {
  const double eps;
  torch::Tensor weight;

public:
  /** Construct an RMSNormImpl module. */
  RMSNormImpl(const int& dim, const double& eps = 1e-6);

  /** Perform root mean square normalization on `x` */
  torch::Tensor _norm(const torch::Tensor& x);
  torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(RMSNorm);

/**
 * Implements rotary positional embeddings.
 * 
 * See here: https://arxiv.org/abs/2104.09864
 */
class RotaryPositionalEmbeddingsImpl : public torch::nn::Module {
  int dim;
  int max_seq_len;
  double base;
  torch::Tensor cache;

public:
  /** Initialize a RotaryPositionalEmbeddingsImpl module. 
   * 
   * @param dim The dimension of the input embeddings.
   * @param max_seq_len The maximum expected sequence length to compute angles for.
   * @param base The base used when computing theta.
  */
  RotaryPositionalEmbeddingsImpl(
    const int& dim, const int& max_seq_len, const double& base = 10000.0
  );

  /** compute and register rope angles as a buffer named cache */
  void init_cache();

  /** Apply the rotary positional embeddings on `x` */
  torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(RotaryPositionalEmbeddings);

class CausalSelfAttentionImpl : public torch::nn::Module {
  ModelConfig config;
  torch::nn::Linear wq;
  torch::nn::Linear wk;
  torch::nn::Linear wv;
  RotaryPositionalEmbeddings rope;
  torch::nn::Linear wo;
  torch::nn::Dropout wo_dropout;

public:
  /** construct a `CausalSelfAttentionImpl` module
   * 
   * @param config The `ModelConfig` POD. 
   */
  CausalSelfAttentionImpl(const ModelConfig& config);

  /** Helper method for registering a linear layer when constructing */
  torch::nn::Linear register_linear(const std::string& name);

  /** Compute and return the attention scores given `x` */
  torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(CausalSelfAttention);

class FeedForwardImpl : public torch::nn::Module {
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
  torch::nn::GELU gelu;

public:
  /** Initialize a `FeedForwardImpl` multi-layer perceptron.
   * 
   * Implements a multi-layer perceptron with `torch::nn::GELU` nonlinearity.
   * 
   * @param dim The input dimension
   */
  FeedForwardImpl(const int& dim);

  /** mlp forward pass. */
  torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(FeedForward);

/** A transformer decoder layer module */
class DecoderLayerImpl : public torch::nn::Module {
  CausalSelfAttention attn;
  RMSNorm attn_norm;
  FeedForward mlp;
  RMSNorm mlp_norm;

public:
  /** Initialize a transformer `DecoderLayer`.
   * 
   * @param config The `ModelConfig` object to initialize modules with.
   */
  DecoderLayerImpl(const ModelConfig& config);
  torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(DecoderLayer);

#endif
