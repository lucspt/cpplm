#include <gtest/gtest.h>
#include <torch/torch.h>
#include <modules.cpp>
#include <iostream>

struct Config {
  int bsz{4};
  int dim{16};
  int seqlen{8};
} config;

class RMSNormTest : public testing::Test {
protected:
  RMSNorm norm_layer;
  torch::Tensor _inps;
  RMSNormTest() : norm_layer(RMSNorm(config.dim)) {};
};

TEST_F(RMSNormTest, ShapeIsNotModified) {
  torch::Tensor inps{torch::rand({config.bsz, config.seqlen, config.dim})};
  EXPECT_EQ(inps.sizes(), norm_layer.forward(inps).sizes());
}