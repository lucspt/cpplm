#include <ATen/TensorIndexing.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/log_normal.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randn.h>
#include <gtest/gtest.h>
#include <model.hpp>
#include <config.hpp>
#include <tuple>

class ModelImplTest : public testing::Test {
public:
  const int VOCAB_SIZE{32};
  const ModelConfig config{VOCAB_SIZE};
  const int BATCH_SIZE{1};
  ModelImpl model;
  std::tuple<torch::Tensor, torch::Tensor> get_tokens_sample() {
    namespace idx = torch::indexing;
    torch::Tensor tokens{
      torch::randint(config.vocab_size, {(config.context_len * BATCH_SIZE) + 1})
    };
    return std::make_tuple(
      tokens.index({idx::Slice{0, -1}}).reshape({BATCH_SIZE, config.context_len}),
      tokens.index({idx::Slice{1, idx::None}}).reshape({BATCH_SIZE, config.context_len})
    );
  }
  ModelImplTest() : model{config} {}
};

TEST_F(ModelImplTest, OuputsVocabSizeNumberOfLogits) {
  const auto& [x, y] = get_tokens_sample();
  EXPECT_EQ(model.forward(x).size(2), VOCAB_SIZE);
};

TEST_F(ModelImplTest, RandomWeightsReturnsExpectedLoss) {
  torch::Tensor expected_loss{-torch::log(torch::tensor(1.0 / VOCAB_SIZE))};
  const auto& [x, y] = get_tokens_sample();
  torch::Tensor loss{model.compute_loss(model.forward(x), y)};
  EXPECT_TRUE(torch::allclose(loss, expected_loss, 1.0));
}
