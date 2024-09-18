#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/options/activation.h>
#include <model.hpp>
#include <modules.hpp>

ModelImpl::ModelImpl(const ModelConfig& config)
  : config{config},
    tok_embedding{config.context_len, config.dim},
    decoder{},
    lm_head{config.dim, config.vocab_size},
    norm{config.dim, config.rmsnorm_eps} {
  tok_embedding = register_module("tok_embedding", tok_embedding);
  for (int n{0}; n < config.n_layers; ++n) {
    decoder->push_back(DecoderLayer{config});
  };
  decoder = register_module("decoder", decoder);
  lm_head = register_module("lm_head", lm_head);
  norm = register_module("norm", norm);
}

torch::Tensor ModelImpl::forward(const torch::Tensor& tokens) {
  torch::Tensor x{tok_embedding->forward(tokens)};
  x = decoder->forward(x);
  x = norm->forward(x);
  x = lm_head->forward(x);
  return x;
};

torch::Tensor ModelImpl::compute_loss(
  const torch::Tensor& logits, const torch::Tensor& target
) {
  torch::IntArrayRef shape{logits.sizes()};  // (bsz, seqlen, embed dim)
  const auto& n{shape[0] * shape[1]};
  return torch::nn::functional::cross_entropy(
    logits.reshape({n, shape[2]}), target.reshape({n})
  );
}
