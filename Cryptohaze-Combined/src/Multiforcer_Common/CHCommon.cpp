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

#include "Multiforcer_Common/CHCommon.h"

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


// file_size: Pass in path, get back file size in bytes.
long int file_size(const char *path) {
  struct stat file_status;

  if(stat(path, &file_status) != 0){
    printf("Unable to stat %s\n", path);
    return -1;
  }
  //printf("File size: %d\n", (int)file_status.st_size);
  return file_status.st_size;
}


int convertAsciiToBinary(char *input, unsigned char *hash, int maxLength) {
  char convertSpace[10];
  uint32_t result;
  int i;

  //TODO: Fix this code to not suck and use scanf
  // Loop until either maxLength is hit, or strlen(intput) / 2 is hit.
  for (i = 0; (i < maxLength) && (i < (strlen(input) / 2)); i++) {
    convertSpace[0] = input[2 * i];
    convertSpace[1] = input[2 * i + 1];
    convertSpace[2] = 0;
    sscanf(convertSpace, "%2x", &result);
    // Do this to prevent scanf from overwriting memory with a 4 byte value...
    hash[i] = (unsigned char)result & 0xff;
  }
  return i;
}


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

int getCudaIsFermi(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    if (deviceProp.major >= 2) {
        return 1;
    } else {
        return 0;
    }
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

// Little utility to remove newlines...
void chomp(char *s) {
    while(*s && *s != '\n' && *s != '\r') s++;
    *s = 0;
}

#ifdef _WIN32
#include <time.h>
#include <windows.h>
#include <iostream>

using namespace std;
#endif
