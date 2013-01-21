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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_SHA1.h"
#include "MFN_Common/MFNDebugging.h"

#define reverse(x)(x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24);

MFNHashTypePlainCUDA_SHA1::MFNHashTypePlainCUDA_SHA1() :  MFNHashTypePlainCUDA(20) {
    trace_printf("MFNHashTypePlainCUDA_SHA1::MFNHashTypePlainCUDA_SHA1()\n");
    this->HashIsBigEndian = 1;
}

void MFNHashTypePlainCUDA_SHA1::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_SHA1::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceNumberStepsToRunPlainSHA1",
        &this->perStep, sizeof(uint32_t));

    //this->printLaunchDebugData(threadData);
    
    MFNHashTypePlainCUDA_SHA1_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

void MFNHashTypePlainCUDA_SHA1::printLaunchDebugData() {
    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);

    printf("Host value passwordLengthPlainSHA1: %d\n", this->passwordLength);
    printf("Host value numberOfHashesPlainSHA1: %lu\n", this->activeHashesProcessed.size());
    printf("Host value deviceGlobalHashlistAddressPlainSHA1: 0x%16x\n", this->DeviceHashlistAddress);
    printf("Host value deviceGlobalBitmapAPlainSHA1: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
    printf("Host value deviceGlobalBitmapBPlainSHA1: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
    printf("Host value deviceGlobalBitmapCPlainSHA1: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
    printf("Host value deviceGlobalBitmapDPlainSHA1: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
    printf("Host value deviceGlobalFoundPasswordsPlainSHA1: 0x%16x\n", this->DeviceFoundPasswordsAddress);
    printf("Host value deviceGlobalFoundPasswordFlagsPlainSHA1: 0x%16x\n", this->DeviceSuccessAddress);
    printf("Host value deviceGlobalStartPointsPlainSHA1: 0x%16x\n", this->DeviceStartPointAddress);
}

std::vector<uint8_t> MFNHashTypePlainCUDA_SHA1::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_SHA1::preProcessHash()\n");
    
    uint32_t *hash32 = (uint32_t *)&rawHash[0];
    
    hash32[0] = reverse(hash32[0]);
    hash32[1] = reverse(hash32[1]);
    hash32[2] = reverse(hash32[2]);
    hash32[3] = reverse(hash32[3]);
    hash32[4] = reverse(hash32[4]);
    
    // After they're in little endian, subtract out the final values.
    hash32[0] -= 0x67452301;
    hash32[1] -= 0xefcdab89;
    hash32[2] -= 0x98badcfe;
    hash32[3] -= 0x10325476;
    hash32[4] -= 0xc3d2e1f0;

    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainCUDA_SHA1::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_SHA1::postProcessHash()\n");
        
    uint32_t *hash32 = (uint32_t *)&processedHash[0];
    
    // Add in the new values before reversing.
    hash32[0] += 0x67452301;
    hash32[1] += 0xefcdab89;
    hash32[2] += 0x98badcfe;
    hash32[3] += 0x10325476;
    hash32[4] += 0xc3d2e1f0;
    

    hash32[0] = reverse(hash32[0]);
    hash32[1] = reverse(hash32[1]);
    hash32[2] = reverse(hash32[2]);
    hash32[3] = reverse(hash32[3]);
    hash32[4] = reverse(hash32[4]);

    return processedHash;
}

void MFNHashTypePlainCUDA_SHA1::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_SHA1::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceCharsetPlainSHA1",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceReverseCharsetPlainSHA1",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("charsetLengthsPlainSHA1",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("constantBitmapAPlainSHA1",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("passwordLengthPlainSHA1",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("numberOfHashesPlainSHA1",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalHashlistAddressPlainSHA1",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalBitmapAPlainSHA1",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalBitmapBPlainSHA1",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalBitmapCPlainSHA1",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalBitmapDPlainSHA1",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalBitmap256kPlainSHA1",
            &this->DeviceBitmap256kb_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalFoundPasswordsPlainSHA1",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainSHA1",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalStartPointsPlainSHA1",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceGlobalStartPasswords32PlainSHA1",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("deviceNumberThreadsPlainSHA1",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_SHA1_CopyValueToConstant("constantBitmapAPlainSHA1", 
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}