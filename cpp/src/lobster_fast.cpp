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

static inline std::string map_event_type(int t) {
  switch (t) {
  case 1: // add
  case 2: // modify
    return "LO+";
  case 3: // cancel
  case 6: // delete
    return "CX+";
  case 4: // execute
  case 5: // execute hidden
  case 7: // trade
    return "MO+";
  default:
    return "LO+";
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(
        stderr,
        "Usage: %s <message_csv> <orderbook_csv> <symbol> [tick_size]\n",
        argv[0]);
    return 1;
  }
  const char *msg_path = argv[1];
  const char *book_path = argv[2];
  const char *symbol = argv[3];
  const double tick = (argc >= 5) ? std::atof(argv[4]) : 0.0;

  std::ifstream msg(msg_path);
  std::ifstream book(book_path);
  if (!msg || !book) {
    std::fprintf(stderr, "Error opening input files.\n");
    return 2;
  }

  std::string mline, bline;
  std::cout << "timestamp,event_type,price,size,level,side,symbol,venue\n";
  while (std::getline(msg, mline) && std::getline(book, bline)) {
    if (mline.empty() || bline.empty())
      continue;
    MessageRow m{};
    if (!parse_message_row(mline, m))
      continue;
    long long ns = static_cast<long long>(m.time * 1e9);
    char side = (m.direction == 1) ? 'B' : 'S';
    double price = m.price;
    if (tick > 0.0) {
      price = std::floor((price / tick) + 0.5) * tick;
    }
    std::cout << ns << "," << map_event_type(m.type) << "," << price << ","
              << m.size << ",," << side << "," << symbol << ",LOBSTER\n";
  }
  return 0;
}
