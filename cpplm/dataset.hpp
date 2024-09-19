#ifndef CPPLM_DATASET_HPP
#define CPPLM_DATASET_HPP
#include <ATen/core/TensorBody.h>
#include <c10/util/string_view.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <filesystem>
#include <string_view>

namespace cpplm {
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

class Dataset : torch::data::datasets::Dataset<Dataset, Example> {
  std::filesystem::directory_iterator dir_iter;
  int batch_size;
  int context_len;
  unsigned long long total_tokens;
  torch::Tensor tokens;

public:
  using Example = Example;

  /** load tokens from a file */
  static inline torch::Tensor load_tokens(c10::string_view f);

  unsigned long long int count_total_tokens();

  explicit Dataset(
    std::string_view dirname,
    std::string_view split,
    const int batch_size,
    const int context_len
  );

  Example get(size_t index) override;

  [[nodiscard]] torch::optional<size_t> size() const override;
};

}  // namespace cpplm
#endif
