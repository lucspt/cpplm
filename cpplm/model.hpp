#include <torch/torch.h>

class Model {
public:
  torch::Tensor compute_loss(const torch::Tensor& logits, const torch::Tensor& target);
  torch::Tensor forward(const torch::Tensor& x);
};