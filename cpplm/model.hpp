#ifndef CPPLM_MODEL_HPP
#define CPPLM_MODEL_HPP

#include <ATen/core/TensorBody.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/pimpl.h>
#include <config.hpp>
#include <modules.hpp>

/**
 * A decoder-only transformer model.
 */
class ModelImpl : public torch::nn::Module {
  ModelConfig config;
  torch::nn::Embedding tok_embedding;
  torch::nn::Sequential decoder;
  torch::nn::Linear lm_head;
  RMSNorm norm;

public:
  /** Initialize a `Model`
 * 
 * @param config The model configuration object.
 */
  ModelImpl(const ModelConfig& config);

  /**
   * Computes the cross entropy loss between `logits` and `targets`
   */
  torch::Tensor compute_loss(const torch::Tensor& logits, const torch::Tensor& target);

  /** Forward pass through the transformer */
  torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(Model);

#endif
