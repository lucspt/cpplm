#include <torch/torch.h>

/**
 * Implements root mean square normalization. 
 * 
 * See here: https://arxiv.org/abs/1910.07467
 */
class RMSNorm : torch::nn::Module {
  double eps;
  torch::Tensor weight;

public:
  /** Construct an RMSNorm module. */
  RMSNorm(const int& dim, const double eps = 1e-6);

  /** Perform root mean square normalization on `x` */
  torch::Tensor _norm(const torch::Tensor& x);
  torch::Tensor forward(const torch::Tensor& x);
};

/**
 * Implements rotary positional embeddings.
 * 
 * See here: https://arxiv.org/abs/2104.09864
 */
class RotaryPositionalEmbeddings : torch::nn::Module {
  int dim;
  int max_seq_len;
  double base;
  torch::Tensor cache;

public:
  /** Initialize a RotaryPositionalEmbeddings module. 
   * 
   * @param dim The dimension of the input embeddings.
   * @param max_seq_len The maximum expected sequence length to compute angles for.
   * @param base The base used when computing theta.
  */
  RotaryPositionalEmbeddings(
    const int& dim, const int& max_seq_len, const double& base = 10000.0
  );

  /** compute and register rope angles as a buffer named cache */
  void init_cache();

  /** Apply the rotary positional embeddings on `x` */
  torch::Tensor forward(const torch::Tensor& x);
};