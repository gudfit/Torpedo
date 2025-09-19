#include <algorithm>
#include <cmath>
#include <numbers>

#include "torpedocode/kernels.hpp"

namespace torpedocode {

namespace {
constexpr double kEps = 1e-12;

torch::Tensor clamp_min_eps(const torch::Tensor &tensor) {
  return torch::clamp_min(tensor, kEps);
}

torch::Tensor make_scalar(double value, const torch::Tensor &like) {
  return torch::full({}, value, like.options());
}
} // namespace

TPPLossOutputs tpp_loss(const TPPLossInputs &inputs) {
  TORCH_CHECK(inputs.intensities.defined(), "intensities tensor must be defined");
  TORCH_CHECK(inputs.intensities.dim() == 3,
              "intensities must have shape [batch, time, num_events]");

  auto intensities = inputs.intensities.contiguous();
  TORCH_CHECK(intensities.dtype() == torch::kFloat32 ||
                  intensities.dtype() == torch::kFloat64,
              "intensities must be float32 or float64");

  const auto B = intensities.size(0);
  const auto T = intensities.size(1);
  const auto M = intensities.size(2);

  TORCH_CHECK(inputs.event_types.defined(), "event_types tensor must be defined");
  TORCH_CHECK(inputs.event_types.dim() == 2,
              "event_types must have shape [batch, time]");
  auto event_types = inputs.event_types.to(torch::kLong).contiguous();
  TORCH_CHECK(event_types.size(0) == B && event_types.size(1) == T,
              "event_types shape must match intensities batch/time");

  TORCH_CHECK(inputs.delta_t.defined(), "delta_t tensor must be defined");
  TORCH_CHECK(inputs.delta_t.dim() == 2, "delta_t must have shape [batch, time]");
  auto delta_t = inputs.delta_t.to(intensities.dtype()).contiguous();
  TORCH_CHECK(delta_t.size(0) == B && delta_t.size(1) == T,
              "delta_t shape must match intensities batch/time");

  auto gather_idx = event_types.clamp_min(0).unsqueeze(-1);
  auto lambda_all = clamp_min_eps(intensities);
  auto lambda_evt = lambda_all.gather(-1, gather_idx).squeeze(-1);
  auto event_mask = (event_types >= 0).to(intensities.options().dtype());
  auto log_lambda_evt = torch::log(lambda_evt) * event_mask;
  auto compensator = (lambda_all.sum(-1) * delta_t).sum(1);

  auto mark_nll = torch::zeros_like(compensator);
  if (inputs.sizes.defined() && inputs.sizes.numel() > 0) {
    TORCH_CHECK(inputs.mark_mu.defined() && inputs.mark_log_sigma.defined(),
                "mark parameters must be provided when sizes tensor is defined");
    TORCH_CHECK(inputs.mark_mu.dim() == 3 && inputs.mark_log_sigma.dim() == 3,
                "mark parameters must have shape [batch, time, num_events]");
    auto mark_mu = inputs.mark_mu.to(intensities.dtype()).contiguous();
    auto mark_log_sigma = inputs.mark_log_sigma.to(intensities.dtype()).contiguous();
    TORCH_CHECK(mark_mu.size(0) == B && mark_mu.size(1) == T && mark_mu.size(2) == M,
                "mark_mu shape must match intensities");
    TORCH_CHECK(mark_log_sigma.size(0) == B && mark_log_sigma.size(1) == T &&
                    mark_log_sigma.size(2) == M,
                "mark_log_sigma shape must match intensities");

    auto sizes = clamp_min_eps(inputs.sizes.to(intensities.dtype()).contiguous());
    TORCH_CHECK(sizes.size(0) == B && sizes.size(1) == T,
                "sizes tensor must match [batch, time]");

    auto mu_evt = mark_mu.gather(-1, gather_idx).squeeze(-1);
    auto log_sigma_evt = mark_log_sigma.gather(-1, gather_idx).squeeze(-1);
    auto sigma = torch::exp(log_sigma_evt);
    auto denom = sigma + kEps;
    auto log_sizes = torch::log(sizes);
    auto z = (log_sizes - mu_evt) / denom;
    const double half_log_two_pi = 0.5 * std::log(2.0 * std::numbers::pi);
    auto constant = make_scalar(half_log_two_pi, intensities);
    auto term = 0.5 * z.pow(2) + log_sigma_evt + log_sizes + constant;
    mark_nll = (term * event_mask).sum(1);
  }

  auto nll = -(log_lambda_evt.sum(1)) + compensator + mark_nll;
  auto nll_mean = nll.mean();

  auto diff = intensities.slice(1, 1) - intensities.slice(1, 0, std::max<int64_t>(T - 1, 0));
  auto diff_sq = diff.pow(2);
  torch::Tensor smoothness;

  const auto norm = inputs.smoothness_norm;
  if (inputs.mask.defined() && inputs.mask.numel() > 0) {
    TORCH_CHECK(inputs.mask.dim() == 2, "mask must have shape [batch, time]");
    auto mask = inputs.mask.to(intensities.dtype()).contiguous();
    TORCH_CHECK(mask.size(0) == B && mask.size(1) == T,
                "mask shape must match [batch, time]");
    auto pair_mask = (mask.slice(1, 1) *
                      mask.slice(1, 0, std::max<int64_t>(T - 1, 0))).unsqueeze(-1);
    auto weighted = diff_sq * pair_mask;
    if (norm == SmoothnessNorm::None) {
      smoothness = weighted.sum();
    } else if (norm == SmoothnessNorm::PerSequence) {
      auto per_seq = weighted.sum({1, 2});
      auto pairs = pair_mask.sum({1, 2}).clamp_min(1.0);
      smoothness = (per_seq / pairs).mean();
    } else {
      auto denom = pair_mask.sum().clamp_min(1.0);
      smoothness = weighted.sum() / denom;
    }
  } else {
    if (norm == SmoothnessNorm::None) {
      smoothness = diff_sq.sum();
    } else if (norm == SmoothnessNorm::PerSequence) {
      auto per_seq = diff_sq.sum({1, 2});
      auto denom_value = static_cast<double>(diff_sq.size(1) * diff_sq.size(2));
      auto denom = make_scalar(denom_value, intensities).clamp_min(1.0);
      smoothness = (per_seq / denom).mean();
    } else {
      smoothness = diff_sq.sum({1, 2}).mean();
    }
  }

  return TPPLossOutputs{nll_mean, smoothness};
}

} // namespace torpedocode
