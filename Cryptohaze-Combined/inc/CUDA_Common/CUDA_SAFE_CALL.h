#ifndef __CUDA_SAFE_CALL
#define __CUDA_SAFE_CALL

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#define CH_CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } \
}

#define CUDA_SAFE_CALL(call) CH_CUDA_SAFE_CALL(call)

#endif