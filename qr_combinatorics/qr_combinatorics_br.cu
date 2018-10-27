#include <stdio.h>
#include <functional>
#include <cstdlib>

#include "qr.h"

#include "qr_read.cuh"
#include "helpers.cuh"

#define THREADS_PER_BLOCK 128
#define BLOCKS 5

#define NUM_ANSWERS 32

__constant__ unsigned char qr_portion[26]; // char size is always 8

__constant__ int qr_size = 0;

__constant__ int qr_width = 0;

__global__ void qr_br(int *first, int *second, int *third, int *fourth){

  int id = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  for(int i = id; i < 8388608 /*2^23*/; i += BLOCKS * THREADS_PER_BLOCK){
    //printf("running id: %i.\n", i);
    unsigned char working[26];
    for(int j = 0; j < 26; j++){
      if(j > 14){
        if(j > 18 && j < 22){
          working[j] = qr_portion[j - 4];
        } else {
          working[j] = 0;
        }
      } else {
        working[j] = qr_portion[j];
      }
    }
    for(int j = 0; 2 + 6 * j < 15; j++){
      working[2 + j * 6] |= ((i >> 4 * j) & 1) << 1;
      working[2 + j * 6] |= ((i >> (4 * j + 1)) & 1);
      if(j * 6 + 3 == 15) continue;
      working[3 + j * 6] |= ((i >> (4 * j + 2)) & 1) << 7;
      working[3 + j * 6] |= ((i >> (4 * j + 3)) & 1) << 6;
    }
    
    working[19] |= ((i >> 10) & 1) << 7;
    for(int j = 0; j < 3; j++){
      for(int k = 0; k < 4; k++){
        working[j + 19] |= ((i >> (10 + j * 4 + k)) & 1) << (6 - 2 * k);
      }
    }

    int bit_idx = 4;
    int mode = extract(working, 0, 4);
    if(mode == 1) { // numeric
      auto x = extract_numeric(working, bit_idx);
      if(x.good){
        atomicAdd(&(first[x.a]), 1);
        atomicAdd(&(second[x.b]), 1);
        atomicAdd(&(third[x.c]), 1);
        atomicAdd(&(fourth[x.d]), 1);
        //bool k = false;
        //for(int j = 0; j < *size; j++){
        //  if(ans == data[j]) k = true;
        //}
        //if(k || size == 0){
        //  continue;
        //}
        //__syncthreads();
        //int index = atomicInc(size, NUM_ANSWERS) - 2;
        //atomicExch(output + index, ans);
        //__syncthreads();
      } else {
        //__syncthreads();
        //__syncthreads();
        //__syncthreads();
      }
    } else {
      //__syncthreads();
      //__syncthreads();
      //__syncthreads();
    }
    // qr_br_full_guess<<<BLOCKS,THREADS_PER_BLOCK>>>(working);
  }
}

void upload(std::array<uint8_t, 18>& data, int width){
  int s = 18;
  cudaMemcpyToSymbol(qr_size, (void*)&s, sizeof(int), 0);
  cudaMemcpyToSymbol(qr_portion, (void**)&data[0], sizeof(uint8_t) * s, 0);
  cudaMemcpyToSymbol(qr_width, (void*)&width, sizeof(int), 0);
}

void wait_for_cuda_end(){
  cudaDeviceReset();
}

void cuda_br(std::array<uint8_t, 18>& data, int width){
  upload(data, width);

  cuda_arr<int, 10> first, second, third, fourth;

  for(int i = 0; i < 10; i++){
    first.host[i] = 0;
    second.host[i] = 0;
    third.host[i] = 0;
    fourth.host[i] = 0;
  }

  first.sync_host();
  second.sync_host();
  third.sync_host();
  fourth.sync_host();

  qr_br<<<BLOCKS, THREADS_PER_BLOCK>>>(first.dev, second.dev, third.dev, fourth.dev);

  first.sync_dev();
  second.sync_dev();
  third.sync_dev();
  fourth.sync_dev();

  printf("First Digit:\n");
  for(int i = 0; i < 10; i++){
    if(first.host[i] == 0) continue;
    printf("%i: %i\n", i, first.host[i]);
  }
  printf("\n");
  printf("Second Digit:\n");
  for(int i = 0; i < 10; i++){
    if(second.host[i] == 0) continue;
    printf("%i: %i\n", i, second.host[i]);
  }
  printf("\n");
  printf("Third Digit:\n");
  for(int i = 0; i < 10; i++){
    if(third.host[i] == 0) continue;
    printf("%i: %i\n", i, third.host[i]);
  }
  printf("\n");
  printf("Fourth Digit:\n");
  for(int i = 0; i < 10; i++){
    if(fourth.host[i] == 0) continue;
    printf("%i: %i\n", i, fourth.host[i]);
  }
  printf("\n\n");
  printf("Possibilities:\n");
  int n = 0;
  for(int i1 = 0; i1 < 10; i1++){
    if(first.host[i1] == 0) continue;
    for(int i2 = 0; i2 < 10; i2++){
      if(second.host[i2] == 0) continue;
      for(int i3 = 0; i3 < 10; i3++){
        if(third.host[i3] == 0) continue;
        for(int i4 = 0; i4 < 10; i4++){
          if(fourth.host[i4] == 0) continue;
          printf("%i%i%i%i\n", i1, i2, i3, i4);
          n++;
        }
      }
    }
  }
  printf("Total Num: %i\n", n);
}
