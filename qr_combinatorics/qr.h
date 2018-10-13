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
  void compute(std::vector<int32_t>, int orientation, int type);
  std::tuple<int, int> check_corner(std::vector<int32_t>, int width);
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
        qr->data[index / 32] &= 0 << (index % 32);
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
    printf("%i", orientation);
    
    internal::compute(d, orientation, cornertype);
#endif
  }
private:
  inline int qr_value_at(int x, int y, int orientation){
    int index = 0;
    if(orientation == 0){ // Top left
      index = (y * (width) + x);
    } else if(orientation == 1){ // Top right
      index = ((width - y - 1) * (width) + x);
    } else if(orientation == 2){ // Bottom left
      index = (y * (width) + (width - x - 1));
    } else { // Bottom right
      index = ((width - y - 1) * (width) + (width - x - 1));
    }
    return (data[index / 32] >> (index % 32)) & 1; // unpack bit from qr_portion
  }
  std::array<int32_t, width> data;
};

#endif
