#include "torpedocode/kernels.hpp"

#include <utility>

namespace torpedocode {

RollingTopoOutputs rolling_topo_cpu(const RollingTopoInputs &inputs) {
  TORCH_CHECK(inputs.series.defined(), "series tensor must be defined");
  TORCH_CHECK(inputs.series.dim() == 2,
              "series must have shape [time, features]");
  TORCH_CHECK(inputs.timestamps.defined(),
              "timestamps tensor must be defined");
  TORCH_CHECK(inputs.timestamps.dim() == 1,
              "timestamps must have shape [time]");
  TORCH_CHECK(inputs.window_sizes.defined(),
              "window_sizes tensor must be defined");
  TORCH_CHECK(inputs.window_sizes.dim() == 1,
              "window_sizes must be a 1D tensor of seconds");
  TORCH_CHECK(inputs.embedding_dim >= 0,
              "embedding_dim must be non-negative");

  auto series = inputs.series.contiguous();
  TORCH_CHECK(!series.is_cuda(),
              "rolling_topo_cpu expects tensors on CPU");
  if (series.dtype() != torch::kFloat32) {
    series = series.to(torch::kFloat32);
  }
  const auto num_rows = series.size(0);
  auto options = series.options().dtype(torch::kFloat32);
  auto embeddings =
      torch::zeros({num_rows, inputs.embedding_dim}, options);
  return RollingTopoOutputs{std::move(embeddings)};
}

RollingTopoOutputs rolling_topo(const RollingTopoInputs &inputs) {
  if (inputs.series.is_cuda()) {
    return rolling_topo_cuda(inputs);
  }
  return rolling_topo_cpu(inputs);
}

} // namespace torpedocode
