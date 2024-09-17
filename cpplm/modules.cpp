#include <torch/torch.h>
#include <modules.hpp>
#include <string>

RMSNormImpl::RMSNormImpl(const int& dim, const double& eps) : eps{eps} {
  weight = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RMSNormImpl::_norm(const torch::Tensor& x) {
  return x * torch::rsqrt(x.pow(2).mean(-1, true) + eps);
  // + eps, don't want to divide by zero when taking reciprocal
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) {
  return weight * _norm(x);
}

RotaryPositionalEmbeddingsImpl::RotaryPositionalEmbeddingsImpl(
  const int& dim, const int& max_seq_len, const double& base
)
  : dim{dim}, max_seq_len{max_seq_len}, base{base} {
  if (dim % 2 != 0) {
    throw std::invalid_argument("`dim` must be divisible by 2");
  }
  init_cache();
}

void RotaryPositionalEmbeddingsImpl::init_cache() {
  torch::Tensor theta{
    1.0
    / torch::pow(
      base,
      torch::arange(0, dim, 2, torch::TensorOptions().dtype(torch::kFloat))
          .index({torch::indexing::Slice(0, dim / 2)})
        / dim
    )
  };
  torch::Tensor m{
    torch::arange(max_seq_len, torch::TensorOptions().dtype(torch::kFloat))
  };
  torch::Tensor freqs{torch::outer(m, theta)};
  // polar for some reason just returns a regular tensor, no `real` and `imag` attributes
  // computing cosine and sin is equivalent.
  freqs = torch::stack({torch::cos(freqs), torch::sin(freqs)}, -1);
  cache = register_buffer("cache", freqs);
}

torch::Tensor RotaryPositionalEmbeddingsImpl::forward(const torch::Tensor& x) {
  // shape of x is (bsz, seqlen, n heads, head dim)
  torch::IntArrayRef sizes{x.sizes()};
  // get angles up until seqlen (seqlen, dim)
  torch::Tensor rope_cache{cache.index({torch::indexing::Slice(0, sizes[1]), "..."})};

  // group into 2 dimensions each (bsz, seqlen, n heads, head dim // 2, 2)
  torch::Tensor xshaped{x.reshape({sizes[0], sizes[1], sizes[2], -1, 2})};
  // (1, seqlen, 1, dim, 2)
  rope_cache = rope_cache.reshape({-1, xshaped.size(1), 1, xshaped.size(3), 2});

  // apply rotations
  torch::Tensor out{torch::stack(
    {xshaped.index({"...", 0}) * rope_cache.index({"...", 0})
       - xshaped.index({"...", 1}) * rope_cache.index({"...", 1}),
     xshaped.index({"...", 1}) * rope_cache.index({"...", 0})
       + xshaped.index({"...", 0}) * rope_cache.index({"...", 1})},
    -1
  )};

  out = out.flatten(3).type_as(x);
  return out;
}

CausalSelfAttentionImpl::CausalSelfAttentionImpl(const ModelConfig& config)
  : config{config},
    wq{torch::nn::LinearOptions(config.dim, config.dim).bias(false)},
    wk{torch::nn::LinearOptions(config.dim, config.dim).bias(false)},
    wv{torch::nn::LinearOptions(config.dim, config.dim).bias(false)},
    rope{config.dim / config.n_heads, config.context_len * 2},
    wo{torch::nn::LinearOptions(config.dim, config.dim).bias(false)},
    wo_dropout{config.dropout} {
  wq = register_module("wq", wq);
  wk = register_module("wk", wk);
  wv = register_module("wv", wv);
  rope = register_module("rope", rope);
  wo = register_module("wo", wo);
  wo_dropout = register_module("wo_dropout", wo_dropout);
}

torch::Tensor CausalSelfAttentionImpl::forward(const torch::Tensor& x) {
  torch::Tensor xq{wq->forward(x)};
  torch::Tensor xk{wk->forward(x)};
  torch::Tensor xv{wv->forward(x)};

  torch::IntArrayRef in_shape{x.sizes()};
  auto b{in_shape[0]}, t{in_shape[1]}, c{in_shape[2]};
  int nhead{config.n_heads};
  xq = xq.reshape({b, t, nhead, c / nhead});
  xk = xk.reshape({b, t, nhead, c / nhead});
  // we can move heads to batch dim now for v, not using rope
  xv = xv.reshape({b, t, nhead, c / nhead}).transpose(1, 2);

  xq = rope->forward(xq).transpose(1, 2);
  xk = rope->forward(xk).transpose(1, 2);

  torch::Tensor y{torch::scaled_dot_product_attention(
    xq, xk, xv, {}, is_training() ? config.dropout : 0.0, true
  )};

  // back to (b, t, c)
  y = y.transpose(1, 2).contiguous().reshape({b, t, c});

  return wo_dropout->forward(wo->forward(y));
}

FeedForwardImpl::FeedForwardImpl(const int& dim)
  : fc1{dim, dim * 4}, fc2{dim * 4, dim}, gelu{} {
  fc1 = register_module("fc1", fc1);
  fc2 = register_module("fc2", fc2);
};

torch::Tensor FeedForwardImpl::forward(const torch::Tensor& x) {
  torch::Tensor xout{fc1->forward(x)};
  xout = gelu->forward(xout);
  xout = fc2->forward(xout);
  return xout;
}

DecoderLayerImpl::DecoderLayerImpl(const ModelConfig& config)
  : attn{config},
    attn_norm{config.dim, config.rmsnorm_eps},
    mlp{config.dim},
    mlp_norm{config.dim, config.rmsnorm_eps} {
  attn = register_module("attn", attn);
  attn_norm = register_module("attn_norm", attn_norm);
  mlp = register_module("mlp", mlp);
  mlp_norm = register_module("mlp_norm", mlp_norm);
}

torch::Tensor DecoderLayerImpl::forward(const torch::Tensor& x) {
  torch::Tensor xout{x + attn->forward(attn_norm->forward(x))};
  xout = xout + mlp->forward(mlp_norm->forward(xout));
  return xout;
}
