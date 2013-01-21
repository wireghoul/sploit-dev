/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2011  Bitweasil (http://www.cryptohaze.com/)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include "CUDA_Common/CHCudaUtils.h"
#include "Multiforcer_Common/CHCommon.h"


void CHCUDAUtils::PrintCUDADeviceInfo(int DeviceId) {
    printf("Should be printing CUDA device info...\n");
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, DeviceId);

    //TODO: Fix me, correct core count.
    printf("CUDA Device Information:\n");
    printf("Device %d: \"%s\"\n", DeviceId, deviceProp.name);
    printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
    printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    printf("  Performance Number:                            %d\n", deviceProp.multiProcessorCount * (deviceProp.clockRate / 1000));
    printf("  RAM (in MB):                                   %d\n", deviceProp.totalGlobalMem / (1024 * 1024));
    printf("  Note: Performance number is clock in mhz * core count, for comparing devices.\n");
    
}


int CHCUDAUtils::ConvertSMVer2Cores(int major, int minor) {
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
          { 0x30, 192},
	  {   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	//printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
    
}


int CHCUDAUtils::getCudaStreamProcessorCount(int CUDADeviceId) {

    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);

    return this->ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
}


int CHCUDAUtils::getCudaHasTimeout(int CUDADeviceId) {
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);
    
    return deviceProp.kernelExecTimeoutEnabled;
}

int CHCUDAUtils::getCudaIsFermi(int CUDADeviceId) {
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);

    if (deviceProp.major >= 2) {
        return 1;
    } else {
        return 0;
    }
}

int CHCUDAUtils::getCudaDefaultThreadCount(int CUDADeviceId) {
    int streamProcCount;
    
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);

    streamProcCount = ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    
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

int CHCUDAUtils::getCudaDefaultBlockCount(int CUDADeviceId) {
    int streamProcCount;
    
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);

    streamProcCount = ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

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

uint64_t CHCUDAUtils::getCudaDeviceGlobalMemory(int CUDADeviceId) {
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);
    
    return deviceProp.totalGlobalMem;
}

uint32_t CHCUDAUtils::getCudaDeviceSharedMemory(int CUDADeviceId) {
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);
    
    return deviceProp.sharedMemPerBlock;
}

int CHCUDAUtils::getCudaIsIntegrated(int CUDADeviceId) {
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);
    
    return deviceProp.integrated;
}


int CHCUDAUtils::getCudaCanMapHostMemory(int CUDADeviceId) {
    cudaDeviceProp deviceProp = this->getDevicePropStruct(CUDADeviceId);
    
    return deviceProp.canMapHostMemory;
}

int CHCUDAUtils::getCudaDeviceCount() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}


cudaDeviceProp CHCUDAUtils::getDevicePropStruct(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    return deviceProp;
}

