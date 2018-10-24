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