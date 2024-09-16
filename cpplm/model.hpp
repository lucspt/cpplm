#include <torch/torch.h>
#include <config.hpp>
class Model : torch::nn::Module {
  ModelConfig config;
  std::shared_ptr<torch::nn::EmbeddingImpl> tok_embedding;
  std::shared_ptr<torch::nn::LinearImpl> lm_head;

public:
  Model(ModelConfig& config);
  torch::Tensor compute_loss(const torch::Tensor& logits, const torch::Tensor& target);
  torch::Tensor forward(const torch::Tensor& x);
};