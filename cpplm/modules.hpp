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