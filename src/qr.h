#ifndef QR_H_
#define QR_H_

#include <functional>
#include <stdint.h>
#include <string>
#include <vector>

#include <iostream>

#include <opencv2/opencv.hpp>

namespace qr {

constexpr static bool black = true;
constexpr static bool white = false;

class qr_t;

struct qr_status {
  std::vector<cv::Mat> contributors_img;
  std::vector<qr_t> contributors_read;
  bool dirtyflag;
  
  std::vector<uint16_t> compute();
};

class qr_t {
public:
  /** \brief Constructor
   \param width number of rows in the qr code (should be same as the number of
   columns)
   \param ver qr code version
   */
  inline qr_t(int width, int ver = 1)
      : m_data([width](){
        int x = 1;
        while(x * 8 < width * width) x++;
        return x;
      }()), // This allocates exactly as much memory (ceiled
                              // to a multiple of 8) as needed.
        m_width(width), m_ver(ver) {}
  /** \breif Default Destructor */
  inline ~qr_t() = default;

  inline qr_t(qr_t &&) = default;
  inline qr_t(const qr_t &) = default;

  inline qr_t &operator=(qr_t &&) = default;
  inline qr_t &operator=(const qr_t &) = default;

  struct value_t {
    int index;
    qr_t *owner;

    /** \brief Assignment operator automates bit packing */
    inline value_t operator=(bool b) {
      if (b) {
        owner->m_data[index / 8] |= (1 << (index % 8));
      } else {
        owner->m_data[index / 8] &= ~(1 << (index % 8));
      }
      return *this;
    }

    /** \brief implicit conversion to bool */
    inline operator bool() {
      return ((owner->m_data[index / 8] >> (index % 8)) & 1) == 1;
    }

    /** \brief implicit conversion to bool */
    inline operator bool() const {
      return ((owner->m_data[index / 8] >> (index % 8)) & 1) == 1;
    }
    
    inline operator int() const {
      return ((owner->m_data[index / 8] >> (index % 8)) & 1) == 1 ? 1 : 0;
    }
  };

  inline value_t operator()(int x, int y) {
    return value_t{y * m_width + x, this};
  }

  inline bool operator()(int x, int y) const {
    return ((m_data[(y * m_width + x) / 8] >> ((y * m_width + x) % 8)) & 1) ==
           1;
  }

  inline int width() { return m_width; }
  inline int width() const { return m_width; }

private:
  friend class computed_qr_t;
  std::vector<uint8_t> m_data;
  int m_width;
  int m_ver;
};


inline std::ostream &operator<<(std::ostream &o, qr::qr_t m_code) {
  for (int i = 0; i < m_code.width(); i++) {
      for (int j = 0; j < m_code.width(); j++) {
        o << (m_code(i, j) ? "\u2588\u2588" : "  ");
      }
      o << "\n";
    }
  return o;
}

enum class computed_qr_orientation_t : uint8_t {
  ROT_0 = 0,
  ROT_90,
  ROT_180,
  ROT_270,
  UNKNOWN,
  NUM_PLUS_ONE
};

enum class computed_qr_type_t : uint8_t {
  BRIDGE_RIGHT = 0,
  BRIDGE_BOTTOM,
  BRIDGE_BOTH,
  CORNERLESS,
  UNKNOWN,
  NUM_PLUS_ONE
};

class computed_qr_t {
public:
  inline computed_qr_t(qr_t &code) : m_code(code) { compute_flags(); }
  inline ~computed_qr_t() = default;
  
  inline computed_qr_t(const computed_qr_t& q) : m_code(q.m_code), m_orientation(q.m_orientation), m_type(q.m_type) {}

  inline computed_qr_t& operator=(const computed_qr_t& q){
    m_code = q.m_code;
    m_orientation = q.m_orientation;
    m_type = q.m_type;
  }
  
  bool test_str(std::string);

  std::vector<uint16_t> compute();

  const qr_t &code() { return m_code; }
  
private:
  qr_t &m_code;
  computed_qr_orientation_t m_orientation = computed_qr_orientation_t::UNKNOWN;
  computed_qr_type_t m_type = computed_qr_type_t::UNKNOWN;
  friend struct qr_status;

  void compute_flags();
};

namespace internal {
std::vector<uint16_t> compute(computed_qr_t bottom);
}

} // namespace qr

inline std::vector<uint16_t> qr::computed_qr_t::compute() {
  if (m_type == qr::computed_qr_type_t::CORNERLESS)
    return qr::internal::compute(*this);

  return {};
}

inline void qr::computed_qr_t::compute_flags() {

  static std::vector<std::function<bool(int, int)>> masks = {
      [](int i, int j) -> bool { return j % 3 == 0; },
      [](int i, int j) -> bool { return (i + j) % 3 == 0; },
      [](int i, int j) -> bool { return (i + j) % 2 == 0; },
      [](int i, int j) -> bool { return i % 2 == 0; },
      [](int i, int j) -> bool { return ((i * j) % 3 + i * j) % 2 == 0; },
      [](int i, int j) -> bool { return ((i * j) % 3 + i + j) % 2 == 0; },
      [](int i, int j) -> bool { return (i / 2 + j / 3) % 2 == 0; },
      [](int i, int j) -> bool { return (i * j) % 2 + (i * j) % 3 == 0; }};

  int current_mask = 0;

  // grabs the rotated code
  auto rotator = [&](int x, int y) -> bool {
    if (m_orientation == qr::computed_qr_orientation_t::ROT_270) {
      return m_code(y, m_code.m_width - 1 - x);
    }
    if (m_orientation == qr::computed_qr_orientation_t::ROT_90) {
      return m_code(m_code.m_width - 1 - y, x);
    }
    if (m_orientation == qr::computed_qr_orientation_t::ROT_180) {
      return m_code(m_code.m_width - 1 - x, m_code.m_width - 1 - y);
    }
    return m_code(x, y);
  };

  auto mask = [&](int x, int y) -> bool {
    return rotator(x, y) ? !masks[current_mask](x, y)
                         : masks[current_mask](x, y);
  };

  auto print = [&](int x = -1, int y = -1) {
    for (int i = 0; i < m_code.m_width; i++) {

      for (int j = 0; j < m_code.m_width; j++) {
        if (i == x && j == y)
          std::cout << (rotator(i, j) ? "!" : ".");
        else
          std::cout << (rotator(i, j) ? "#" : " ");
      }
      std::cout << std::endl;
    }
  };

  bool found_corner = false;
  for (int i = 0;
       i < static_cast<int>(qr::computed_qr_orientation_t::NUM_PLUS_ONE) - 1;
       i++) {
    m_orientation = static_cast<qr::computed_qr_orientation_t>(i);
    bool found_square = [&]() {
      for (int x = 0; x < 7; x++) {
        for (int y = 0; y < 7; y++) {
          bool m = ((x == 1 || x == 5 || y == 1 || y == 5) && x != 0 &&
                    y != 0 && x != 6 && y != 6)
                       ? white
                       : black;
          if (rotator(x, y) != m) {
            return false;
          }
        }
      }
      return true;
    }();
    if (found_square) {
      found_corner = true;
      break;
    }
    m_orientation = qr::computed_qr_orientation_t::UNKNOWN;
  }
  if (found_corner) {
    // Search for bridges
    bool right = true, bottom = true;
    for (int i = 7; i < m_code.m_width; i++) {
      if (rotator(i, 6) == (i % 2 == 0 ? white : black))
        right = false;
      if (rotator(6, i) == (i % 2 == 0 ? white : black))
        bottom = false;
    }
    if (right && bottom)
      m_type = qr::computed_qr_type_t::BRIDGE_BOTH;
    else if (right)
      m_type = qr::computed_qr_type_t::BRIDGE_RIGHT;
    else if (bottom)
      m_type = qr::computed_qr_type_t::BRIDGE_BOTTOM;
    else
      std::cout << "ERROR: corner does not have bridges!" << std::endl;
  } else {
    m_type = qr::computed_qr_type_t::CORNERLESS;
  }
}

inline bool qr::computed_qr_t::test_str(std::string str) {
  if (str == "tl" && m_type == qr::computed_qr_type_t::BRIDGE_BOTH)
    return true;
  if (str == "tr" && m_type == qr::computed_qr_type_t::BRIDGE_BOTTOM)
    return true;
  if (str == "bl" && m_type == qr::computed_qr_type_t::BRIDGE_RIGHT)
    return true;
  if (str == "br" && m_type == qr::computed_qr_type_t::CORNERLESS)
    return true;
  return false;
}

void localize(cv::Mat &, cv::Mat &, qr::qr_status& );

inline std::vector<uint16_t> qr::qr_status::compute(){
  std::vector<qr::computed_qr_t> computed;
  computed.reserve(contributors_read.size());
  for(int i = 0; i < contributors_read.size(); i++){
    computed.emplace_back(contributors_read[i]);
  }
  std::vector<uint16_t> values;
  for(int i = 0; i < computed.size(); i++){
    auto& q = computed[i];
    if(q.m_type == qr::computed_qr_type_t::CORNERLESS){
      auto list = q.compute();
      for(int j = 0; j < list.size(); j++){
        values.push_back(list[j]);
      }
      computed.erase(computed.begin() + i); // Remove all bottom right corners for future loops
      i--;
    }
  }
  for(int i = 0; i < computed.size(); i++){
    //TODO other corner types
  }
}

#endif
