#include <gtest/gtest.h>
#include <torch/torch.h>
#include <modules.cpp>
#include <iostream>
#include <config.hpp>

struct Config : public ModelConfig {
  int bsz{4};
} config;

// #################################################################
// RMSNorm
class RMSNormTest : public testing::Test {
protected:
  RMSNorm norm_layer;
  RMSNormTest() : norm_layer{RMSNorm(config.dim)} {};
};

TEST_F(RMSNormTest, ShapeIsNotModified) {
  torch::Tensor inps{torch::rand({config.bsz, config.context_len, config.dim})};
  EXPECT_EQ(inps.sizes(), norm_layer.forward(inps).sizes());
}

// #################################################################
// RotaryPositionalEmbedding

TEST(RotaryPositionalEmbeddings, RaisesIfDimNotEven) {
  EXPECT_THROW(
    RotaryPositionalEmbeddings(11, config.context_len * 2), std::invalid_argument
  );
}
class RotaryPositionalEmbeddingsTest : public testing::Test {
protected:
  const torch::Tensor inps;
  RotaryPositionalEmbeddings rope_layer;
  RotaryPositionalEmbeddingsTest()
    : rope_layer{RotaryPositionalEmbeddings(
        config.dim / config.n_heads,  // expects head dim, not dim
        config.context_len * 2
      )},
      inps{torch::randn(
        {config.bsz, config.context_len, config.n_heads, config.dim / config.n_heads}
      )} {};
};

TEST_F(RotaryPositionalEmbeddingsTest, ShapeIsNotModified) {
  EXPECT_EQ(rope_layer.forward(inps).sizes(), inps.sizes());
};

TEST_F(RotaryPositionalEmbeddingsTest, FirstRotationHasNoEffect) {
  torch::Tensor out{rope_layer.forward(inps)};
  auto nbatch{inps.size(0)};
  for (int i{0}; i < nbatch; ++i) {
    ASSERT_TRUE(torch::allclose(inps.index({i, 0}), out.index({i, 0})));
  };
}