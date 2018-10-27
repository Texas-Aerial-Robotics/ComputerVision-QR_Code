#ifndef HELPERS_H_
#define HELPERS_H_

template <typename T, int S>
struct cuda_arr {
    constexpr static int size = S;
    T *host;
    T *dev;
    inline cuda_arr(){
        host = (T*) malloc(sizeof(T) * S);
        cudaMalloc((void**) &dev, sizeof(T) * S);
    }
    inline ~cuda_arr(){
        free(host);
        cudaFree(dev);
    }
    inline void sync_host(){
        cudaMemcpy(dev, host, sizeof(T) * S, cudaMemcpyHostToDevice);
    }
    inline void sync_dev(){
        cudaMemcpy(host, dev, sizeof(T) * S, cudaMemcpyDeviceToHost);
    }
};

#endif