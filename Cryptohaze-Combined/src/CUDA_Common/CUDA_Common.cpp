#include "CUDA_Common/CUDA_Common.h"

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>

// Beginning of GPU Architecture definitions
inline int ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{ { 0x10,  8 },
	  { 0x11,  8 },
	  { 0x12,  8 },
	  { 0x13,  8 },
	  { 0x20, 32 },
	  { 0x21, 48 },
	  {   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}
// end of GPU Architecture definitions



void printCudaDeviceInfo(int deviceId) {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    printf("CUDA Device Information:\n");
    printf("Device %d: \"%s\"\n", deviceId, deviceProp.name);
    printf("  Integrated:                                    %d\n", deviceProp.integrated);
    printf("  Can map host mem:                              %d\n", deviceProp.canMapHostMemory);
    printf("  Number of cores:                               %d\n",
        ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    printf("  Performance Number:                            %d\n",
        ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount *
            (deviceProp.clockRate / 1000));
    printf("  Note: Performance number is clock in mhz * core count, for comparing devices.\n");
}

uint64_t getCudaDeviceMemoryBytes(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    return (uint64_t)deviceProp.totalGlobalMem;
}

int getCudaStreamProcessorCount(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    return ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
}

int getCudaHasTimeout(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    return deviceProp.kernelExecTimeoutEnabled;
}

int getCudaDefaultThreadCountBySPCount(int streamProcCount) {
    if (streamProcCount < 128) {
        // Low end GPUs: < 128 stream processors.
        return 192;
    } else if (streamProcCount > 256) {
        // High end GPUs: Fermis.
        return 512;
    } else {
        // Moderate GPUs - Teslas
        return 256;
    }
}

int getCudaDefaultBlockCountBySPCount(int streamProcCount) {
    if (streamProcCount < 128) {
        // Low end GPUs: < 128 stream processors.
        return 60;
    } else if (streamProcCount > 256) {
        // High end GPUs: Fermis.
        return 240;
    } else {
        // Moderate GPUs - Teslas
        return 120;
    }
}