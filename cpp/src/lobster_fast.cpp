#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct MessageRow {
  double time; // seconds since start of day
  int type;
  long long order_id;
  double size;
  double price;
  int direction; // 1=buy, -1=sell
};

static inline bool parse_message_row(const std::string &line, MessageRow &out) {
  std::stringstream ss(line);
  std::string cell;
  // time
  if (!std::getline(ss, cell, ','))
    return false;
  out.time = std::atof(cell.c_str());
  // type
  if (!std::getline(ss, cell, ','))
    return false;
  out.type = std::atoi(cell.c_str());
  // order_id
  if (!std::getline(ss, cell, ','))
    return false;
  out.order_id = std::atoll(cell.c_str());
  // size
  if (!std::getline(ss, cell, ','))
    return false;
  out.size = std::atof(cell.c_str());
  // price
  if (!std::getline(ss, cell, ','))
    return false;
  out.price = std::atof(cell.c_str());
  // direction
  if (!std::getline(ss, cell, ','))
    return false;
  out.direction = std::atoi(cell.c_str());
  return true;
}

static inline std::string map_event_type(int t, int direction) {
  // direction: 1=buy (bid), -1=sell (ask)
  const char sign = (direction == -1) ? '-' : '+';
  switch (t) {
  case 1: // add
  case 2: // modify
    return std::string("LO") + sign;
  case 3: // cancel
  case 6: // delete
    return std::string("CX") + sign;
  case 4: // execute
  case 5: // execute hidden
  case 7: // trade
    return std::string("MO") + sign;
  default:
    return std::string("LO") + sign;
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "Usage: %s <message_csv> <orderbook_csv> <symbol> [tick_size] "
                 "[session_date YYYY-MM-DD] [tz_offset_seconds]\n",
                 argv[0]);
    return 1;
  }
  const char *msg_path = argv[1];
  const char *book_path = argv[2];
  const char *symbol = argv[3];
  const double tick = (argc >= 5) ? std::atof(argv[4]) : 0.0;
  std::string session_date = (argc >= 6) ? std::string(argv[5]) : std::string();
  long tz_offset_seconds = (argc >= 7) ? std::atol(argv[6]) : 0L;

  std::ifstream msg(msg_path);
  std::ifstream book;
  bool use_book = true;
  if (std::string(book_path) != "-") {
    book.open(book_path);
    if (!book) {
      use_book = false;
    }
  } else {
    use_book = false;
  }
  if (!msg) {
    std::fprintf(stderr, "Error opening message input file.\n");
    return 2;
  }

  // Preload orderbook snapshots (times + price vectors per side)
  std::vector<double> book_times;
  std::vector<std::vector<double>> ask_prices;
  std::vector<std::vector<double>> bid_prices;
  int Ldet = 0;
  if (use_book) {
    std::string header;
    if (!std::getline(book, header)) {
      use_book = false;
    }
    if (use_book) {
      std::vector<std::string> cols;
      std::stringstream ss(header);
      std::string c;
      while (std::getline(ss, c, ','))
        cols.push_back(c);
      int time_col = -1;
      std::vector<int> ask_idx;
      std::vector<int> bid_idx;
      for (size_t i = 0; i < cols.size(); ++i) {
        if (cols[i] == std::string("time") ||
            cols[i] == std::string("timestamp"))
          time_col = (int)i;
        if (cols[i].rfind("ask_price_", 0) == 0)
          ask_idx.push_back((int)i);
        if (cols[i].rfind("bid_price_", 0) == 0)
          bid_idx.push_back((int)i);
      }
      std::sort(ask_idx.begin(), ask_idx.end());
      std::sort(bid_idx.begin(), bid_idx.end());
      Ldet = (int)std::min(ask_idx.size(), bid_idx.size());
      std::string line;
      while (std::getline(book, line)) {
        if (line.empty())
          continue;
        std::stringstream sb(line);
        std::string cell;
        int idx = 0;
        double tval = book_times.empty() ? 0.0 : book_times.back();
        std::vector<double> ap(Ldet, 0.0), bp(Ldet, 0.0);
        int ai = 0, bi = 0;
        while (std::getline(sb, cell, ',')) {
          if (idx == time_col) {
            tval = std::atof(cell.c_str());
          }
          if (ai < Ldet && idx == ask_idx[ai]) {
            ap[ai] = std::atof(cell.c_str());
            ai++;
          }
          if (bi < Ldet && idx == bid_idx[bi]) {
            bp[bi] = std::atof(cell.c_str());
            bi++;
          }
          idx++;
        }
        book_times.push_back(tval);
        ask_prices.push_back(std::move(ap));
        bid_prices.push_back(std::move(bp));
      }
      if (book_times.empty())
        use_book = false;
    }
  }

  std::string mline;
  std::cout << "timestamp,event_type,price,size,level,side,symbol,venue\n";
  size_t book_pos = 0;
  while (std::getline(msg, mline)) {
    if (mline.empty())
      continue;
    MessageRow m{};
    if (!parse_message_row(mline, m))
      continue;
    long long ns = static_cast<long long>(m.time * 1e9);
    if (!session_date.empty()) {
      // Convert seconds since local midnight to UTC epoch ns using provided
      // date and offset Very light parser for YYYY-MM-DD
      int y = 0, mo = 0, d = 0;
      if (std::sscanf(session_date.c_str(), "%d-%d-%d", &y, &mo, &d) == 3) {
        std::tm t = {};
        t.tm_year = y - 1900;
        t.tm_mon = mo - 1;
        t.tm_mday = d;
        t.tm_hour = 0;
        t.tm_min = 0;
        t.tm_sec = 0;
        // timegm: convert UTC tm to epoch seconds; fallback to timegm if
        // available
#ifdef _GNU_SOURCE
        time_t base_s = timegm(&t);
#else
        // Portable fallback: use mktime as local and adjust by timezone, but
        // may be off; acceptable as best-effort.
        time_t base_s = timegm(&t);
#endif
        if (base_s > 0) {
          long long base_ns = static_cast<long long>(base_s) * 1000000000LL;
          ns = base_ns -
               (static_cast<long long>(tz_offset_seconds) * 1000000000LL) + ns;
        }
      }
    }
    char side = (m.direction == 1) ? 'B' : 'S';
    double price = m.price;
    if (tick > 0.0) {
      price = std::floor((price / tick) + 0.5) * tick;
    }
    // Stateful level inference from preloaded snapshots
    int level = 0;
    if (use_book && Ldet > 0) {
      while (book_pos + 1 < book_times.size() &&
             book_times[book_pos + 1] <= m.time) {
        book_pos++;
      }
      auto approx_eq = [&](double a, double b) {
        double tol = (tick > 0.0) ? (0.5 * tick) : 1e-9;
        return std::fabs(a - b) <= tol;
      };
      if (side == 'S') {
        for (int l = 1; l <= Ldet; ++l) {
          double ap = ask_prices[book_pos][l - 1];
          if (approx_eq(ap, price)) {
            level = l;
            break;
          }
        }
      } else {
        for (int l = 1; l <= Ldet; ++l) {
          double bp = bid_prices[book_pos][l - 1];
          if (approx_eq(bp, price)) {
            level = l;
            break;
          }
        }
      }
    }

    std::cout << ns << "," << map_event_type(m.type, m.direction) << ","
              << price << "," << m.size << ","
              << (level > 0 ? std::to_string(level) : std::string("")) << ","
              << side << "," << symbol << ",LOBSTER\n";
  }
  return 0;
}
