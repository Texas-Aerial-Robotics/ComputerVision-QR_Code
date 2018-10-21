#include <stdio.h>
#include <functional>
#include <cstdlib>

#include "qr.h"

#define THREADS_PER_BLOCK 128

__constant__ int qr_portion[32]; // int type is always 32 bits, max size should be 32*8 bits

__constant__ int qr_size = 0;

__global__ void qr_compute(void){
  int id = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  printf("running id: %i, size is %i.\n", id, qr_size);


}

void compute_qr_possibilities1(std::vector<int32_t> data){
  int s = data.size();
  cudaMemcpyToSymbol(qr_size, (void*)&s, sizeof(int), 0);
  cudaMemcpyToSymbol(qr_portion, (void**)&data[0], sizeof(int) * s, 0);
  qr_compute <<< 1, THREADS_PER_BLOCK >>> ();
}

void wait_for_cuda_end(){
  cudaDeviceReset();
}

std::vector<uint16_t> qr_tl(std::vector<int32_t>, int, internal::orientation_t);
std::vector<uint16_t> qr_tr(std::vector<int32_t>, int, internal::orientation_t);
std::vector<uint16_t> qr_bl(std::vector<int32_t>, int, internal::orientation_t);
std::vector<uint16_t> qr_br(std::vector<int32_t>, int, internal::orientation_t);

std::vector<uint16_t> internal::compute(std::vector<int32_t> data, int width, internal::orientation_t orientation, internal::corner_type_t cornertype){
  if(cornertype == internal::corner_type_t::top_left){
    return qr_tl(data, width, orientation);
  } else if(cornertype == internal::corner_type_t::top_right) {
    return qr_tr(data, width, orientation);
  } else if(cornertype == internal::corner_type_t::bottom_left) {
    return qr_bl(data, width, orientation);
  } else {
    return qr_br(data, width, orientation);
  }
}

inline int qr_value_at_m(int y, int x, int orientation, int width, std::vector<int32_t>& data){
  int index = 0;
  if(orientation == 0){ // Top left
    index = (y * (width) + x);
  } else if(orientation == 1){ // Top right
    index = (x * width + (width - y - 1));
  } else if(orientation == 2){ // Bottom right
    index = ((width - y - 1) * (width) + (width - x - 1));
  } else { // Bottom left
    index = ((width - x - 1) * width + y);
  }
  return (data[index / 32] >> (index % 32)) & 1; // unpack bit from qr_portion
}

inline int qr_value_set_m(int y, int x, int orientation, int width, std::vector<int32_t>& data, bool d){
  int index = 0;
  if(orientation == 0){ // Top left
    index = (y * (width) + x);
  } else if(orientation == 1){ // Top right
    index = (x * width + (width - y - 1));
  } else if(orientation == 2){ // Bottom right
    index = ((width - y - 1) * (width) + (width - x - 1));
  } else { // Bottom left
    index = ((width - x - 1) * width + y);
  }
  if(d)
    data[index / 32] |= 1 << (index % 32);
  else
    data[index / 32] &= ~(1 << (index % 32));
}

/* https://en.wikipedia.org/wiki/QR_code#/media/File:QR_Format_Information.svg */
std::vector<std::function<bool(int, int)> > masks = {
  [](int i, int j) -> bool { // 0
    return j % 3 == 0;
  },[](int i, int j) -> bool { // 1
    return (i + j) % 3 == 0;
  },[](int i, int j) -> bool { // 2
    return (i + j) % 2 == 0;
  },[](int i, int j) -> bool { // 3
    return i % 2 == 0;
  },[](int i, int j) -> bool { // 4
    return ((i * j) % 3 + i * j) % 2 == 0;
  },[](int i, int j) -> bool { // 5
    return ((i * j) % 3 + i + j) % 2 == 0;
  },[](int i, int j) -> bool { // 6
    return (i / 2 + i / 3) % 2 == 0;
  },[](int i, int j) -> bool { // 7
    return (i * j) % 2 + (i * j) % 3 == 0;
  }
};

std::vector<uint16_t> qr_tl(std::vector<int32_t> d, int width, internal::orientation_t o){
  int k = static_cast<uint8_t>(o);
  HELPER1(m, width, d)
  HELPER2
  int x = (black(2, 8, k) ? 4 : 0) + (black(3, 8, k) ? 2 : 0) + (black(4, 8, k) ? 1 : 0);
  for(int i = 0; i < width; i++){
    for(int j = 0; j < width; j++){
      if(masks[x](i,j)) {
        qr_value_set_m(j, i, k, width, d, black(i, j, k));
      }
    }
  }
  /* https://en.wikipedia.org/wiki/QR_code#/media/File:QR_Character_Placement.svg */
  std::vector<uint8_t> known;
  //TODO
}

std::vector<uint16_t> qr_tr(std::vector<int32_t> d, int width, internal::orientation_t o){
  //int k = static_cast<uint8_t>(o);
  HELPER1(m, width, d)
  HELPER2
  //TODO
}

std::vector<uint16_t> qr_bl(std::vector<int32_t> d, int width, internal::orientation_t o){
  //int k = static_cast<uint8_t>(o);
  HELPER1(m, width, d)
  HELPER2
  //TODO
}

std::vector<uint16_t> qr_br(std::vector<int32_t> d, int width, internal::orientation_t o){
  HELPER1(m, width, d)
  HELPER2
  std::vector<std::tuple<int, int> > possibilities;
  for(int k = 0; k < 4; k++) { // We really don't know what orientation this is
    
    for(int m = 0; m < masks.size(); m++){
      //TODO
    }
    
  }
}
