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

#include "MFN_CUDA_host/MFNHashTypeSaltedCUDA_IPBWL.h"
#include "MFN_CUDA_device/MFN_CUDA_ConstantCopyDefines.h"
#include "MFN_Common/MFNDebugging.h"

// NOTE: This constructor has to call all the way down!  I don't know what happens
// if you use different sizes for different classes - please don't do it!
MFNHashTypeSaltedCUDA_IPBWL::MFNHashTypeSaltedCUDA_IPBWL() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypeSaltedCUDA_IPBWL::MFNHashTypeSaltedCUDA_IPBWL()\n");
    
    this->hashAttributes.hashWordWidth32 = 1;
    this->hashAttributes.hashUsesSalt = 1;
    this->hashAttributes.hashUsesWordlist = 1;
}

void MFNHashTypeSaltedCUDA_IPBWL::launchKernel() {
    trace_printf("MFNHashTypeSaltedCUDA_IPBWL::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_STEPS_TO_RUN,
        &this->perStep, sizeof(uint32_t));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_STARTING_SALT_OFFSET,
        &this->saltStartOffset, sizeof(uint32_t));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_START_STEP,
        &this->startStep, sizeof(uint32_t));
    
    MFNHashTypeSaltedCUDA_IPBWL_LaunchKernel(this->wordlistBlockLength, this->GPUBlocks, this->GPUThreads);
}

void MFNHashTypeSaltedCUDA_IPBWL::printLaunchDebugData() {
    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);

    printf("Host value passwordLengthPlainMD5: %d\n", this->passwordLength);
    printf("Host value numberOfHashesPlainMD5: %lu\n", this->activeHashesProcessed.size());
    printf("Host value deviceGlobalHashlistAddressPlainMD5: 0x%16x\n", this->DeviceHashlistAddress);
    printf("Host value deviceGlobalBitmapAPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
    printf("Host value deviceGlobalBitmapBPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
    printf("Host value deviceGlobalBitmapCPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
    printf("Host value deviceGlobalBitmapDPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
    printf("Host value deviceGlobalFoundPasswordsPlainMD5: 0x%16x\n", this->DeviceFoundPasswordsAddress);
    printf("Host value deviceGlobalFoundPasswordFlagsPlainMD5: 0x%16x\n", this->DeviceSuccessAddress);
    printf("Host value deviceGlobalStartPointsPlainMD5: 0x%16x\n", this->DeviceStartPointAddress);
}




void MFNHashTypeSaltedCUDA_IPBWL::copyConstantDataToDevice() {
    trace_printf("MFNHashTypeSaltedCUDA_IPBWL::copyConstantDataToDevice()\n");
    cudaError_t err;

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    // Begin copying constant data to the device.

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_CONSTANT_BITMAP_A,
            &this->sharedBitmap8kb_a[0], 8192);

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_PASSWORD_LENGTH,
            &localPasswordLength, sizeof(uint8_t));

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_NUMBER_OF_HASHES,
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_HASHLIST_ADDRESS,
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_A,
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_B,
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_C,
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_D,
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_256KB_A,
            &this->DeviceBitmap256kb_Address, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_FOUND_PASSWORDS,
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_GLOBAL_FOUND_PASSWORD_FLAGS,
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_NUMBER_THREADS,
            &localNumberThreads, sizeof(uint64_t));
    
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_WORDLIST_DATA,
            &this->DeviceWordlistBlocks, sizeof(uint32_t *));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_WORDLIST_LENGTHS,
            &this->DeviceWordlistLengths, sizeof(uint8_t *));
    
    this->copySaltConstantsToDevice();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error %d: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, err, cudaGetErrorString( err));
        exit(1);
    }
}

void MFNHashTypeSaltedCUDA_IPBWL::copySaltConstantsToDevice() {
    trace_printf("MFNHashTypeSaltedCUDA_IPBWL::copySaltConstantsToDevice()\n");
    cudaError_t err;
    // Salted hash data
    
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_SALT_LENGTHS,
            &this->DeviceSaltLengthsAddress, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_SALT_DATA,
            &this->DeviceSaltValuesAddress, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_NUMBER_SALTS,
            &this->numberSaltsCopiedToDevice, sizeof(this->numberSaltsCopiedToDevice));
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}

void MFNHashTypeSaltedCUDA_IPBWL::copyWordlistSizeToDevice(uint32_t wordCount, uint8_t blocksPerWord) {
    trace_printf("MFNHashTypeSaltedCUDA_IPBWL::copyWordlistSizeToDevice()\n");
    cudaError_t err;

    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_NUMBER_WORDS,
            &wordCount, sizeof(wordCount));
    MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(MFN_CUDA_DEVICE_BLOCKS_PER_WORD,
            &blocksPerWord, sizeof(blocksPerWord));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}
