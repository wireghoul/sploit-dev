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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_LOTUS.h"
#include "MFN_Common/MFNDebugging.h"
#include "CUDA_Common/CHCudaUtils.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"

extern MFNClassFactory MultiforcerGlobalClassFactory;

MFNHashTypePlainCUDA_LOTUS::MFNHashTypePlainCUDA_LOTUS() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypePlainCUDA_LOTUS::MFNHashTypePlainCUDA_LOTUS()\n");
}

void MFNHashTypePlainCUDA_LOTUS::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_LOTUS::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceNumberStepsToRunPlainLOTUS",
        &this->perStep, sizeof(uint32_t));

    //this->printLaunchDebugData(threadData);
    
    MFNHashTypePlainCUDA_LOTUS_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

uint32_t MFNHashTypePlainCUDA_LOTUS::getMaxHardwareThreads(uint32_t requestedThreadCount) {
    CHCUDAUtils *CudaUtils = MultiforcerGlobalClassFactory.getCudaUtilsClass();
    uint32_t sharedMemSize;
    uint32_t maxThreadCountByMem;
    
    sharedMemSize = CudaUtils->getCudaDeviceSharedMemory(this->gpuDeviceId);
    
    // If shared mem == 16kb, max of 64 threads, else max of 512 (for 48kb).
    if (sharedMemSize <= (16*1024)) {
        maxThreadCountByMem = 64;
    } else {
        maxThreadCountByMem = 512;
    }
    
    // If the requested amount is beyond the supported max, return max.
    if (requestedThreadCount > maxThreadCountByMem) {
        return maxThreadCountByMem;
    }
    return requestedThreadCount;
}

void MFNHashTypePlainCUDA_LOTUS::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_LOTUS::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceCharsetPlainLOTUS",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceReverseCharsetPlainLOTUS",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("charsetLengthsPlainLOTUS",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("constantBitmapAPlainLOTUS",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("passwordLengthPlainLOTUS",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("numberOfHashesPlainLOTUS",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalHashlistAddressPlainLOTUS",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalBitmapAPlainLOTUS",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalBitmapBPlainLOTUS",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalBitmapCPlainLOTUS",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalBitmapDPlainLOTUS",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalBitmap256kPlainLOTUS",
            &this->DeviceBitmap256kb_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalFoundPasswordsPlainLOTUS",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainLOTUS",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalStartPointsPlainLOTUS",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceGlobalStartPasswords32PlainLOTUS",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("deviceNumberThreadsPlainLOTUS",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant("constantBitmapAPlainLOTUS", 
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}