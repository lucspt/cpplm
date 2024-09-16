#include <torch/torch.h>
#include <model.hpp>

torch::Tensor Model::forward(const torch::Tensor& x) {
  return torch::tensor(1);
};

torch::Tensor Model::compute_loss(
  const torch::Tensor& logits, const torch::Tensor& target
) {
  torch::Tensor t{target.reshape({-1})};
  torch::IntArrayRef shape{logits.sizes()};
  const long& batches{shape[0] * shape[1]};
  torch::Tensor l{logits.reshape({batches, shape[2]})};
  return torch::nn::functional::cross_entropy(l, t);
}