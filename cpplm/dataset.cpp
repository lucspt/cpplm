#include <ATen/core/TensorBody.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/string_view.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/linear.h>
#include <cstddef>
#include <dataset.hpp>
#include <filesystem>
#include <functional>
#include <optional>
#include <tuple>

cpplm::Dataset::Dataset(
  std::string_view dirname,
  std::string_view split,
  const int batch_size,
  const int context_len
)
  : dir_iter{std::format("{}/{}", dirname, split)},
    batch_size{batch_size},
    context_len{context_len},
    total_tokens{count_total_tokens()} {};

inline torch::Tensor cpplm::Dataset::load_tokens(c10::string_view f) {
  torch::Tensor t{torch::from_file(f, std::nullopt, 0, torch::kLong)};
  return t;
}

unsigned long long cpplm::Dataset::count_total_tokens() {
  long long count{};
  for (const auto& d : dir_iter) {
    count += load_tokens(d.path().string()).size(0);
  }
  return count;
}

torch::optional<size_t> cpplm::Dataset::size() const {
  return total_tokens;
};

cpplm::Dataset::Example cpplm::Dataset::get(size_t _) {
  std::function<void(int)>{[](int) -> void { return; }};

  return {torch::rand({2, 21}), torch::rand({2, 21})};
}
