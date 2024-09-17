#include <gtest/gtest.h>
#include <torch/torch.h>
#include <modules.hpp>
#include <iostream>
#include <config.hpp>

const ModelConfig model_config{};
struct Config : public ModelConfig {
  int bsz{4};
} config;

// #################################################################
// RMSNormImpl
class RMSNormImplTest : public testing::Test {
protected:
  RMSNormImpl norm_layer;
  RMSNormImplTest() : norm_layer{RMSNormImpl(config.dim)} {};
};

TEST_F(RMSNormImplTest, ShapeIsNotModified) {
  torch::Tensor inps{torch::rand({config.bsz, config.context_len, config.dim})};
  EXPECT_EQ(inps.sizes(), norm_layer.forward(inps).sizes());
}

// #################################################################
// RotaryPositionalEmbedding

TEST(RotaryPositionalEmbeddingsImpl, RaisesIfDimNotEven) {
  EXPECT_THROW(
    RotaryPositionalEmbeddingsImpl(11, config.context_len * 2), std::invalid_argument
  );
}
class RotaryPositionalEmbeddingsImplTest : public testing::Test {
protected:
  const torch::Tensor inps;
  RotaryPositionalEmbeddingsImpl rope_layer;
  RotaryPositionalEmbeddingsImplTest()
    : rope_layer{RotaryPositionalEmbeddingsImpl(
        config.dim / config.n_heads,  // expects head dim, not dim
        config.context_len * 2
      )},
      inps{torch::randn(
        {config.bsz, config.context_len, config.n_heads, config.dim / config.n_heads}
      )} {};
};

TEST_F(RotaryPositionalEmbeddingsImplTest, ShapeIsNotModified) {
  EXPECT_EQ(rope_layer.forward(inps).sizes(), inps.sizes());
};

TEST_F(RotaryPositionalEmbeddingsImplTest, FirstRotationHasNoEffect) {
  torch::Tensor out{rope_layer.forward(inps)};
  auto nbatch{inps.size(0)};
  for (int i{0}; i < nbatch; ++i) {
    ASSERT_TRUE(torch::allclose(inps.index({i, 0}), out.index({i, 0})));
  };
}

// #################################################################
// CausalSelfAttentionImpl

class CausalSelfAttentionImplTest : public testing::Test {
protected:
  torch::Tensor inps;
  CausalSelfAttentionImpl attn_layer;

  CausalSelfAttentionImplTest()
    : inps{torch::randn({config.bsz, config.context_len, config.dim})},
      attn_layer{CausalSelfAttentionImpl{ModelConfig{}}} {};
};

TEST_F(CausalSelfAttentionImplTest, ShapeIsNotModified) {
  EXPECT_EQ(attn_layer.forward(inps).sizes(), inps.sizes());
}

// #################################################################
// DecoderLayer

class DecoderLayerImplTest : public testing::Test {
protected:
  DecoderLayerImpl d_layer;
  torch::Tensor inps;

  DecoderLayerImplTest()
    : d_layer{DecoderLayerImpl{model_config}},
      inps{torch::rand({config.bsz, config.context_len, config.dim})} {}
};

TEST_F(DecoderLayerImplTest, ShapeIsNotModified) {
  EXPECT_EQ(d_layer.forward(inps).sizes(), inps.sizes());
}
