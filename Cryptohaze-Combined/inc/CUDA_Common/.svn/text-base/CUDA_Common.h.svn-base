#ifndef _CUDA_COMMON_H
#define	_CUDA_COMMON_H

#ifdef _WIN32
#include "windows/stdint.h"
#else
#include <stdint.h>
#endif

void printCudaDeviceInfo(int deviceId);

// Returns the number of stream processors a device has for basic auto-tune.
int getCudaStreamProcessorCount(int deviceId);

// Returns true if the device has a timeout set.
int getCudaHasTimeout(int deviceId);

// Get the RAM off the GPU
uint64_t getCudaDeviceMemoryBytes(int deviceId);

// Get the default thread & block counts
int getCudaDefaultThreadCountBySPCount(int streamProcCount);
int getCudaDefaultBlockCountBySPCount(int streamProcCount);


#endif	/* _CUDA_COMMON_H */

