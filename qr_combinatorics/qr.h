#ifndef QR_H_
#define QR_H_

#include <bitset>
#include <array>
#include <vector>
#include <stdio.h>
#include <tuple>

#define QR_LINE 29

#define QR_SIZE QR_LINE*QR_LINE

#define QR_PORTION_SIZE (QR_SIZE / 4)

namespace internal {
  enum class orientation_t : uint8_t {
    deg_0 = 0,
    deg_90,
    deg_180,
    deg_270
  };
  enum class corner_type_t : uint8_t {
    top_left = 0,
    top_right,
    bottom_right,
    bottom_left
  };
  //void compute(std::vector<int32_t>, orientation_t orientation, corner_type_t type);
  std::tuple<orientation_t, corner_type_t> check_corner(std::vector<int32_t>, int width);
  std::vector<uint16_t> compute(std::vector<int32_t>, int, orientation_t, corner_type_t);
}

template <int width>
class qr_comb_t {
public:
  inline qr_comb_t(){
    for(int i = 0; i < width; i++){
      data[i] = 0; // sets all values to zeroes to ensure no issues
    }
  }
  
  inline ~qr_comb_t() {
    
  }
  
  struct value {
    qr_comb_t *qr;
    int index;
    
    inline operator bool(){
      return ((qr->data[index / 32] >> (index % 32)) & 1) == 1;
    }
    
    inline bool operator=(bool b){
      if(b)
        qr->data[index / 32] |= 1 << (index % 32);
      else
        qr->data[index / 32] &= ~(1 << (index % 32));
      return b;
    }
  };
  
  inline value operator()(int x, int y){
    return value {this, y * width + x};
  }
  
  inline void compute(){
    std::vector<int32_t> d;
    d.reserve(width);
    for(int i = 0; i < data.size(); i++){
      
      d.push_back(data[i]);
    }
    // Precompute
#ifndef __NVCC__ // nvcc doesn't support structured bindings (as far as I can tell)
    auto [orientation, cornertype] = internal::check_corner(d, width);
    
    /*return*/ internal::compute(d, width, orientation, cornertype);
#endif
    //return {};
  }
private:
  std::array<int32_t, width> data;
};

#define HELPER1(m, width, d) auto qr_value_at = [&](int x, int y, int orientation){ \
    return qr_value_at_##m(x, y, orientation, width, d); \
  }; \
  auto black = [&](int y, int x, int k, bool m = true){ \
    return qr_value_at(x, y, k) != (m ? 1 : 0); \
  };
  
#define HELPER2 auto printerr = [&](int y, int x, int k){\
    for(int i = 0; i < width; i++){\
      for(int j = 0; j < width; j++){\
        if(i == x && j == y) printf((qr_value_at(i, j, k) == 1) ? "!" : "|");\
        else printf((qr_value_at(i, j, k) == 1) ? "." : " ");\
      }\
      printf("\n");\
    }\
  };

#endif
