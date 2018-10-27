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
    int memsize = (size * size) / 8 + 1; // Number of bytes to store
    data.reserve(memsize);

    for(int i = 0; i < memsize; i++){
      data[i] = 0; // Initialize all values to zero
    }
  }
  
  inline ~qr_comb_t() {}
  
  /* Helper to facilitate value assignment */
  struct value {
    qr_comb_t *qr;
    int index;
    
    inline operator bool(){
      return qr->extract(index, 0);
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
  
  // Value getter/setter
  inline value operator()(int x, int y){
    return value {this, y * width + x};
  }

  // Gets value of qr code at coords (x,y)
  inline uint8_t extract(int x, int y){
    int index = y * width + x;
    return (data[index / 8] >> (index % 8)) & 1;
  }
  
  void compute(); // Precomputation
private:
  int width; // width of qr code (width x width is the size)
  std::vector<uint8_t> data;
};

#endif
