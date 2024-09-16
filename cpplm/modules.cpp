#include <torch/torch.h>
#include <iostream>
#include <modules.hpp>

RMSNorm::RMSNorm(const int& dim, const double eps) : eps(eps) {
  weight = register_parameter("weight", torch::ones(dim));
}

torch::Tensor RMSNorm::_norm(const torch::Tensor& x) {
  return x * torch::rsqrt(x.pow(2).mean(-1, true) + eps);
  // + eps, don't want to divide by zero when taking reciprocal
}

torch::Tensor RMSNorm::forward(const torch::Tensor& x) {
  return weight * _norm(x);
}

RotaryPositionalEmbeddings::RotaryPositionalEmbeddings(
  const int& dim, const int& max_seq_len, const double& base
)
  : dim{dim}, max_seq_len{max_seq_len}, base{base} {
  if (dim % 2 != 0) {
    throw std::invalid_argument("`dim` must be divisible by 2");
  }
  init_cache();
}

void RotaryPositionalEmbeddings::init_cache() {
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

torch::Tensor RotaryPositionalEmbeddings::forward(const torch::Tensor& x) {
  // shape of x is (bsz, seqlen, n heads, head dim)
  torch::IntArrayRef sizes{x.sizes()};
  // get angles up until seqlen (seqlen, dim)
  torch::Tensor rope_cache{cache.index({torch::indexing::Slice(0, sizes[1]), "..."})};

  std::cout << rope_cache.sizes() << '\n' << rope_cache[0] << '\n';

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
