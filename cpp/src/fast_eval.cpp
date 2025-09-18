#include <algorithm>
#include <cmath>
#include <cstdint>
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

struct Metrics { double auroc, auprc, brier, ece; };

static Metrics compute_metrics(const std::vector<double>& pred, const std::vector<int>& label) {
  const size_t n = pred.size();
  // AUROC via Mann–Whitney U
  std::vector<size_t> order(n); for (size_t i=0;i<n;++i) order[i]=i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b){return pred[a] < pred[b];});
  std::vector<size_t> rank(n);
  for (size_t i=0;i<n;++i) rank[order[i]] = i;
  size_t n_pos=0, n_neg=0; for (auto y: label) (y? n_pos: n_neg)++;
  double auc = NAN;
  if (n_pos>0 && n_neg>0) {
    double sum_ranks_pos=0.0; for (size_t i=0;i<n;++i) if (label[i]==1) sum_ranks_pos += rank[i];
    auc = (sum_ranks_pos - n_pos*(n_pos-1)/2.0) / (double(n_pos)*double(n_neg));
  }
  // AUPRC by trapezoid on PR curve
  std::vector<size_t> o2(n); for (size_t i=0;i<n;++i) o2[i]=i;
  std::stable_sort(o2.begin(), o2.end(), [&](size_t a, size_t b){return pred[a] > pred[b];});
  std::vector<int> y_sorted(n); for (size_t i=0;i<n;++i) y_sorted[i] = label[o2[i]];
  std::vector<double> tp(n), fp(n);
  for (size_t i=0;i<n;++i) {
    tp[i] = (i? tp[i-1]:0.0) + (y_sorted[i]==1 ? 1.0:0.0);
    fp[i] = (i? fp[i-1]:0.0) + (y_sorted[i]==0 ? 1.0:0.0);
  }
  double P = double(n_pos>0?n_pos:1);
  std::vector<double> precision(n), recall(n);
  for (size_t i=0;i<n;++i) { double denom = (tp[i]+fp[i]); precision[i] = denom>0? tp[i]/denom : 0.0; recall[i] = tp[i]/P; }
  double auprc = 0.0; for (size_t i=1;i<n;++i) { double dx = recall[i]-recall[i-1]; auprc += 0.5*(precision[i]+precision[i-1])*dx; }
  // Brier (vectorized accumulation when possible)
  double brier=0.0;
#if defined(__AVX2__)
  {
    size_t i = 0;
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
      __m256d p = _mm256_set_pd(pred[i+3], pred[i+2], pred[i+1], pred[i+0]);
      __m256d yv = _mm256_set_pd((double)label[i+3], (double)label[i+2], (double)label[i+1], (double)label[i+0]);
      __m256d d = _mm256_sub_pd(p, yv);
      acc = _mm256_fmadd_pd(d, d, acc);
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc);
    brier = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) { double d = pred[i] - label[i]; brier += d*d; }
  }
#else
  for (size_t i=0;i<n;++i) { double d = pred[i]-label[i]; brier += d*d; }
#endif
  brier /= (n>0?n:1);
  // ECE with 15 equal-frequency bins
  const int M=15;
  std::vector<size_t> ord(n); for (size_t i=0;i<n;++i) ord[i]=i;
  std::stable_sort(ord.begin(), ord.end(), [&](size_t a,size_t b){return pred[a] < pred[b];});
  size_t base = n/M; size_t rem = n%M;
  std::vector<size_t> starts(M), sizes(M);
  {
    size_t s=0; for (int m=0;m<M;++m){ size_t sz = base + (m<rem?1:0); starts[m]=s; sizes[m]=sz; s+=sz; }
  }
  double ece=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:ece)
#endif
  for (int m=0; m<M; ++m) {
    size_t sz = sizes[m];
    if (sz==0) continue;
    size_t st = starts[m];
    double sum_p=0.0, sum_y=0.0;
    for (size_t j=0;j<sz;++j){ size_t idx = ord[st+j]; sum_p += pred[idx]; sum_y += label[idx]; }
    double conf = sum_p/double(sz); double acc = sum_y/double(sz);
    ece += (double(sz)/double(n)) * std::abs(acc-conf);
  }
  return Metrics{auc, auprc, brier, ece};
}

int main(int argc, char** argv) {
  int argi = 1;
  int threads = -1;
  // Optional: --threads N
  while (argi + 1 < argc && std::string(argv[argi]) == "--threads") {
    threads = std::atoi(argv[argi + 1]);
    argi += 2;
  }
#ifdef _OPENMP
  if (threads > 0) {
    omp_set_num_threads(threads);
  }
#endif
  if (argc - argi < 2) {
    std::fprintf(stderr, "Usage: %s [--threads N] <predictions_csv> <output_json>\n", argv[0]);
    return 1;
  }
  std::ifstream in(argv[argi + 0]);
  if (!in) { std::fprintf(stderr, "Cannot open %s\n", argv[1]); return 2; }
  std::string line; std::getline(in, line); // header
  int pred_col=-1, label_col=-1; {
    std::stringstream ss(line); std::string cell; int idx=0;
    while (std::getline(ss, cell, ',')) {
      if (cell=="pred") pred_col=idx; if (cell=="label") label_col=idx; idx++;
    }
  }
  if (pred_col<0 || label_col<0) { std::fprintf(stderr, "CSV must have pred,label columns\n"); return 3; }
  std::vector<double> pred; std::vector<int> label;
  while (std::getline(in, line)) {
    if (line.empty()) continue; std::stringstream ss(line); std::string cell; int idx=0; double p=0.0; int y=0;
    while (std::getline(ss, cell, ',')) { if (idx==pred_col) p=std::atof(cell.c_str()); if (idx==label_col) y=std::atoi(cell.c_str()); idx++; }
    pred.push_back(p); label.push_back(y);
  }
  Metrics m = compute_metrics(pred, label);
  std::ofstream out(argv[argi + 1]);
  out << "{\n";
  out << "  \"auroc\": " << (std::isfinite(m.auroc)? m.auroc : 0.0) << ",\n";
  out << "  \"auprc\": " << m.auprc << ",\n";
  out << "  \"brier\": " << m.brier << ",\n";
  out << "  \"ece\": " << m.ece << "\n";
  out << "}\n";
  return 0;
}
