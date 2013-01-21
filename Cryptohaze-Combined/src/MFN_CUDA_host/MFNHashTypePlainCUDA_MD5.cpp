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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_MD5.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_CUDA_device/MFN_CUDA_ConstantCopyDefines.h"

MFNHashTypePlainCUDA_MD5::MFNHashTypePlainCUDA_MD5() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypePlainCUDA_MD5::MFNHashTypePlainCUDA_MD5()\n");
    
    memset(&this->hashAttributes, 0, sizeof(this->hashAttributes));
    // 32-bit words in the hash.
    this->hashAttributes.hashWordWidth32 = 1;
    // 128 bits total - create a-d bitmaps
    this->hashAttributes.create128mbBitmapA = 1;
    this->hashAttributes.create128mbBitmapB = 1;
    this->hashAttributes.create128mbBitmapC = 1;
    this->hashAttributes.create128mbBitmapD = 1;

    // Only need 256kb A bitmap for MD5
    this->hashAttributes.create256kbBitmapA = 1;
    
    // Create A shared bitmaps for various sizes
    this->hashAttributes.create32kbBitmapA = 1;
    this->hashAttributes.create16kbBitmapA = 1;
    this->hashAttributes.create8kbBitmapA = 1;
}

void MFNHashTypePlainCUDA_MD5::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_MD5::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_DEVICE_STEPS_TO_RUN,
        &this->perStep, sizeof(uint32_t));

    //this->printLaunchDebugData(threadData);
    
    MFNHashTypePlainCUDA_MD5_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

void MFNHashTypePlainCUDA_MD5::printLaunchDebugData() {
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

std::vector<uint8_t> MFNHashTypePlainCUDA_MD5::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_MD5::preProcessHash()\n");
    this->MD5.prepareHash(this->passwordLength, rawHash);
    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainCUDA_MD5::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_MD5::postProcessHash()\n");
    this->MD5.postProcessHash(this->passwordLength, processedHash);
    return processedHash;
}

void MFNHashTypePlainCUDA_MD5::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_MD5::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Local variables to match device sizes
    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_DEVICE_CHARSET_FORWARD,
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_DEVICE_CHARSET_REVERSE,
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_DEVICE_CHARSET_LENGTHS,
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_CONSTANT_BITMAP_A,
            &this->sharedBitmap8kb_a[0], 8192);

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_DEVICE_PASSWORD_LENGTH,
            &localPasswordLength, sizeof(uint8_t));

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_NUMBER_OF_HASHES,
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_HASHLIST_ADDRESS,
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_A,
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_B,
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_C,
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_D,
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_BITMAP_256KB_A,
            &this->DeviceBitmap256kb_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_FOUND_PASSWORDS,
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_GLOBAL_FOUND_PASSWORD_FLAGS,
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_DEVICE_START_PASSWORDS,
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_MD5_CopyValueToConstantById(MFN_CUDA_DEVICE_NUMBER_THREADS,
            &localNumberThreads, sizeof(uint64_t));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}