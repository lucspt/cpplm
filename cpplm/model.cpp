#include <torch/torch.h>
#include <model.hpp>

Model::Model(ModelConfig& config)
  : config{config},
    tok_embedding{register_module(
      "tok_embedding", torch::nn::Embedding(config.context_len, config.dim)
    )},
    lm_head{register_module("lm_head", torch::nn::Linear(config.dim, config.dim))} {};

torch::Tensor Model::forward(const torch::Tensor& tokens) {
  torch::Tensor x{tok_embedding->forward(tokens)};
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
