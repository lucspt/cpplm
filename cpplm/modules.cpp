#include <torch/torch.h>
#include <modules.hpp>

RMSNorm::RMSNorm(const int& dim, const double eps) : eps(eps) {
  weight = register_parameter("weight", torch::ones(dim));
}

torch::Tensor RMSNorm::_norm(const torch::Tensor& x) {
  return x * torch::rsqrt(x.pow(2).mean(-1, true) + eps);
  // + eps, don't want to divide by zero when taking reciprocal
}

torch::Tensor RMSNorm::forward(const torch::Tensor& x) {
  return weight * _norm(x);
}
