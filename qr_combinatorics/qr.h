#ifndef QR_H_
#define QR_H_

#include <bitset>
#include <array>
#include <vector>
#include <stdio.h>
#include <tuple>

constexpr static bool black = true;
constexpr static bool white = false;

class qr_comb_t {
public:
  inline qr_comb_t(int size) : width(size) {
    int memsize = (size * size) / 8 + 1;
    data.reserve(memsize);

    for(int i = 0; i < memsize; i++){
      data[i] = 0;
    }
  }
  
  inline ~qr_comb_t() {}
  
  struct value {
    qr_comb_t *qr;
    int index;
    
    inline operator bool(){
      return false; // TODO
    }
    
    inline bool operator=(bool b) && {
      if(b){
        qr->data[index / 8] |= (1 << (index % 8));
      } else {
        qr->data[index / 8] &= ~(1 << (index % 8));
      }
      return b;
    }
  };
  
  inline value operator()(int x, int y){
    return value {this, y * width + x};
  }

  inline uint8_t extract(int x, int y){
    int index = y * width + x;
    return (data[index / 8] >> (index % 8)) & 1;
  }
  
  void compute();
private:
  int width;
  std::vector<uint8_t> data;
};

#endif
