#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__)
#pragma GCC optimize("O3", "unroll-loops", "fast-math")
#endif

struct Metrics {
  double auroc, auprc, brier, ece;
};

static Metrics compute_metrics(const std::vector<double> &pred,
                               const std::vector<int> &label,
                               bool pr_step_mode) {
  const size_t n = pred.size();
  // Single sort (desc) reused for AUROC, AUPRC, ECE to reduce overhead
  std::vector<size_t> o2(n);
  for (size_t i = 0; i < n; ++i)
    o2[i] = i;
  std::sort(o2.begin(), o2.end(),
            [&](size_t a, size_t b) { return pred[a] > pred[b]; });
  std::vector<int> y_sorted(n);
  for (size_t i = 0; i < n; ++i)
    y_sorted[i] = label[o2[i]];
  size_t n_pos = 0, n_neg = 0;
  for (auto y : label)
    (y ? n_pos : n_neg)++;
  double auc = NAN;
  if (n_pos > 0 && n_neg > 0) {
    // AUROC via U-statistic from desc order: sum_ranks_pos(asc) = n_pos*(n-1) -
    // sum(pos_desc_indices)
    double sum_pos_desc_idx = 0.0;
    for (size_t i = 0; i < n; ++i)
      if (y_sorted[i] == 1)
        sum_pos_desc_idx += double(i);
    double sum_ranks_pos_asc = double(n_pos) * double(n - 1) - sum_pos_desc_idx;
    auc = (sum_ranks_pos_asc - double(n_pos) * double(n_pos - 1) / 2.0) /
          (double(n_pos) * double(n_neg));
  }
  // AUPRC: either trapezoid (trap) or step-wise (sklearn-like)
  std::vector<double> tp(n), fp(n);
  for (size_t i = 0; i < n; ++i) {
    tp[i] = (i ? tp[i - 1] : 0.0) + (y_sorted[i] == 1 ? 1.0 : 0.0);
    fp[i] = (i ? fp[i - 1] : 0.0) + (y_sorted[i] == 0 ? 1.0 : 0.0);
  }
  double P = double(n_pos > 0 ? n_pos : 1);
  std::vector<double> precision(n), recall(n);
  for (size_t i = 0; i < n; ++i) {
    double denom = (tp[i] + fp[i]);
    precision[i] = denom > 0 ? tp[i] / denom : 0.0;
    recall[i] = tp[i] / P;
  }
  double auprc = 0.0;
  if (pr_step_mode) {
    // Step-wise: sum (R_i - R_{i-1}) * precision_i (right-constant)
    for (size_t i = 1; i < n; ++i) {
      double dx = recall[i] - recall[i - 1];
      if (dx > 0)
        auprc += precision[i] * dx;
    }
  } else {
    for (size_t i = 1; i < n; ++i) {
      double dx = recall[i] - recall[i - 1];
      auprc += 0.5 * (precision[i] + precision[i - 1]) * dx;
    }
  }
  // Brier (vectorized accumulation when possible)
  auto accumulate_range = [&](size_t start, size_t end) {
    double local = 0.0;
#if defined(__AVX2__)
    size_t i = start;
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= end; i += 4) {
      __m256d p =
          _mm256_set_pd(pred[i + 3], pred[i + 2], pred[i + 1], pred[i + 0]);
      __m256d yv = _mm256_set_pd((double)label[i + 3], (double)label[i + 2],
                                 (double)label[i + 1], (double)label[i + 0]);
      __m256d d = _mm256_sub_pd(p, yv);
      acc = _mm256_fmadd_pd(d, d, acc);
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc);
    local += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < end; ++i) {
      double d = pred[i] - label[i];
      local += d * d;
    }
#else
    for (size_t i = start; i < end; ++i) {
      double d = pred[i] - label[i];
      local += d * d;
    }
#endif
    return local;
  };

  double brier = 0.0;
#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
  if (max_threads > 1) {
#pragma omp parallel
    {
      size_t T = (size_t)omp_get_num_threads();
      size_t t = (size_t)omp_get_thread_num();
      size_t start = (n * t) / T;
      size_t end = (n * (t + 1)) / T;
      double local = accumulate_range(start, end);
#pragma omp atomic
      brier += local;
    }
  } else {
    brier = accumulate_range(0, n);
  }
#else
  brier = accumulate_range(0, n);
#endif
  brier /= (n > 0 ? n : 1);
  // ECE with 15 equal-frequency bins (reuse desc order; direction doesn't
  // affect bins)
  const int M = 15;
  size_t base = n / M;
  size_t rem = n % M;
  std::vector<size_t> starts(M), sizes(M);
  {
    size_t s = 0;
    for (int m = 0; m < M; ++m) {
      size_t sz = base + (m < rem ? 1 : 0);
      starts[m] = s;
      sizes[m] = sz;
      s += sz;
    }
  }
  double ece = 0.0;
  for (int m = 0; m < M; ++m) {
    size_t sz = sizes[m];
    if (sz == 0)
      continue;
    size_t st = starts[m];
    double sum_p = 0.0, sum_y = 0.0;
    for (size_t j = 0; j < sz; ++j) {
      size_t idx = o2[st + j];
      sum_p += pred[idx];
      sum_y += label[idx];
    }
    double conf = sum_p / double(sz);
    double acc = sum_y / double(sz);
    ece += (double(sz) / double(n)) * std::abs(acc - conf);
  }
  return Metrics{auc, auprc, brier, ece};
}

int main(int argc, char **argv) {
  // Minor I/O tuning (mostly harmless for ifstream but cheap)
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  int argi = 1;
  int threads = -1;
  bool pr_step_mode = false; // default trapezoid
  // Optional: --threads N
  while (argi < argc) {
    std::string a = argv[argi];
    if (a == "--threads" && argi + 1 < argc) {
      threads = std::atoi(argv[argi + 1]);
      argi += 2;
      continue;
    }
    if (a == "--pr-mode" && argi + 1 < argc) {
      std::string mode = argv[argi + 1];
      pr_step_mode = (mode == "sklearn" || mode == "step");
      argi += 2;
      continue;
    }
    break;
  }
#ifdef _OPENMP
  if (threads > 0) {
    omp_set_num_threads(std::max(1, threads));
    omp_set_dynamic(0);
  }
#endif
  if (argc - argi < 2) {
    std::fprintf(stderr,
                 "Usage: %s [--threads N] [--pr-mode trap|sklearn] "
                 "<predictions_csv> <output_json>\n",
                 argv[0]);
    return 1;
  }
  std::ifstream in(argv[argi + 0]);
  if (!in) {
    std::fprintf(stderr, "Cannot open %s\n", argv[1]);
    return 2;
  }
  std::string line;
  std::getline(in, line); // header
  int pred_col = -1, label_col = -1;
  {
    std::stringstream ss(line);
    std::string cell;
    int idx = 0;
    while (std::getline(ss, cell, ',')) {
      if (cell == "pred")
        pred_col = idx;
      if (cell == "label")
        label_col = idx;
      idx++;
    }
  }
  if (pred_col < 0 || label_col < 0) {
    std::fprintf(stderr, "CSV must have pred,label columns\n");
    return 3;
  }
  std::vector<double> pred;
  pred.reserve(65536);
  std::vector<int> label;
  label.reserve(65536);
  const bool fast_path =
      (line == "idx,pred,label" && pred_col == 1 && label_col == 2);
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    if (fast_path) {
      // Fast parse: use rfind to locate last two commas and parse substrings
      size_t last = line.rfind(',');
      if (last == std::string::npos) {
        continue;
      }
      size_t prev = (last > 0) ? line.rfind(',', last - 1) : std::string::npos;
      if (prev == std::string::npos) {
        continue;
      }
      std::string pred_s = line.substr(prev + 1, last - (prev + 1));
      std::string label_s = line.substr(last + 1);
      double p = std::atof(pred_s.c_str());
      int y = std::atoi(label_s.c_str());
      if (std::isfinite(p) && (y == 0 || y == 1)) {
        pred.push_back(p);
        label.push_back(y);
      }
      continue;
    }
    // Generic parse
    std::stringstream ss(line);
    std::string cell;
    int idx = 0;
    double p = 0.0;
    int y = 0;
    while (std::getline(ss, cell, ',')) {
      if (idx == pred_col)
        p = std::atof(cell.c_str());
      if (idx == label_col)
        y = std::atoi(cell.c_str());
      idx++;
    }
    if (std::isfinite(p) && (y == 0 || y == 1)) {
      pred.push_back(p);
      label.push_back(y);
    }
  }
  Metrics m = compute_metrics(pred, label, pr_step_mode);
  std::ofstream out(argv[argi + 1]);
  out << "{\n";
  out << "  \"auroc\": " << (std::isfinite(m.auroc) ? m.auroc : 0.0) << ",\n";
  out << "  \"auprc\": " << m.auprc << ",\n";
  out << "  \"brier\": " << m.brier << ",\n";
  out << "  \"ece\": " << m.ece << "\n";
  out << "}\n";
  return 0;
}
