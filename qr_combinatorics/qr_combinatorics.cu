#include <stdio.h>
#include <bitset>
#include <cstdlib>

#include "qr.h"

#define THREADS_PER_BLOCK 128

__constant__ int qr_portion[32]; // int type is always 32 bits, max size should be 32*8 bits

__constant__ int qr_size = 0;

__device__ void qr_top_left(){
  printf("top_left\n");
  
}

__device__ void qr_top_right(){
  printf("top_right\n");
}

__device__ void qr_bottom_left(){
  printf("bottom_left\n");
}

__device__ void qr_bottom_right(){
  printf("bottom_right\n");
}

__device__ int qr_value_at(int x, int y, int orientation){
  int index = 0;
  if(orientation == 0){ // Top left
    index = (y * (qr_size) + x);
  } else if(orientation == 1){ // Top right
    index = ((qr_size - y - 1) * (qr_size) + x);
  } else if(orientation == 2){ // Bottom left
    index = (y * (qr_size) + (qr_size - x - 1));
  } else { // Bottom right
    index = ((qr_size - y - 1) * (qr_size) + (qr_size - x - 1));
  }
  return (qr_portion[index / 32] >> (index % 32)) & 1; // unpack bit from qr_portion
}

__device__ void printerr(int x, int y, int k) {
    for(int i = 0; i < qr_size; i++){
      for(int j = 0; j < qr_size; j++){
        if(i == x && j == y) printf("!");
        else printf((qr_value_at(i, j, k) == 1) ? "." : " ");
      }
      printf("\n");
    }
}

__global__ void qr_compute(void){
  int id = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  printf("running id: %i, size is %i.\n", id, qr_size);


}

void compute_qr_possibilities1(std::vector<int32_t> data){
  int s = data.size();
  cudaMemcpyToSymbol(qr_size, (void*)&s, sizeof(int), 0);
  cudaMemcpyToSymbol(qr_portion, (void**)&data[0], sizeof(int) * s, 0);
  qr_compute <<<1, 1>>> ();
}

void wait_for_cuda_end(){
  cudaDeviceReset();
}

void internal::compute(std::vector<int32_t> data, int, int){
  //compute_qr_possibilities1(data);
}
