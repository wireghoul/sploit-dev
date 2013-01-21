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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_SHA256.h"
#include "MFN_Common/MFNDebugging.h"

#define reverse(x)(x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24);

MFNHashTypePlainCUDA_SHA256::MFNHashTypePlainCUDA_SHA256() :  MFNHashTypePlainCUDA(32) {
    trace_printf("MFNHashTypePlainCUDA_SHA256::MFNHashTypePlainCUDA_SHA256()\n");
    this->HashIsBigEndian = 1;
}

void MFNHashTypePlainCUDA_SHA256::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_SHA256::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceNumberStepsToRunPlainSHA256",
        &this->perStep, sizeof(uint32_t));

    //this->printLaunchDebugData(threadData);
    
    MFNHashTypePlainCUDA_SHA256_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

void MFNHashTypePlainCUDA_SHA256::printLaunchDebugData() {
    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);

    printf("Host value passwordLengthPlainSHA256: %d\n", this->passwordLength);
    printf("Host value numberOfHashesPlainSHA256: %lu\n", this->activeHashesProcessed.size());
    printf("Host value deviceGlobalHashlistAddressPlainSHA256: 0x%16x\n", this->DeviceHashlistAddress);
    printf("Host value deviceGlobalBitmapAPlainSHA256: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
    printf("Host value deviceGlobalBitmapBPlainSHA256: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
    printf("Host value deviceGlobalBitmapCPlainSHA256: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
    printf("Host value deviceGlobalBitmapDPlainSHA256: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
    printf("Host value deviceGlobalFoundPasswordsPlainSHA256: 0x%16x\n", this->DeviceFoundPasswordsAddress);
    printf("Host value deviceGlobalFoundPasswordFlagsPlainSHA256: 0x%16x\n", this->DeviceSuccessAddress);
    printf("Host value deviceGlobalStartPointsPlainSHA256: 0x%16x\n", this->DeviceStartPointAddress);
}

std::vector<uint8_t> MFNHashTypePlainCUDA_SHA256::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_SHA256::preProcessHash()\n");
    
    uint32_t *hash32 = (uint32_t *)&rawHash[0];
    
    hash32[0] = reverse(hash32[0]);
    hash32[1] = reverse(hash32[1]);
    hash32[2] = reverse(hash32[2]);
    hash32[3] = reverse(hash32[3]);
    hash32[4] = reverse(hash32[4]);
    hash32[5] = reverse(hash32[5]);
    hash32[6] = reverse(hash32[6]);
    hash32[7] = reverse(hash32[7]);
    
    // After they're in little endian, subtract out the final values.
//    hash32[0] -= 0x67452301;
//    hash32[1] -= 0xefcdab89;
//    hash32[2] -= 0x98badcfe;
//    hash32[3] -= 0x10325476;
//    hash32[4] -= 0xc3d2e1f0;

    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainCUDA_SHA256::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_SHA256::postProcessHash()\n");
        
    uint32_t *hash32 = (uint32_t *)&processedHash[0];
    
    // Add in the new values before reversing.
//    hash32[0] += 0x67452301;
//    hash32[1] += 0xefcdab89;
//    hash32[2] += 0x98badcfe;
//    hash32[3] += 0x10325476;
//    hash32[4] += 0xc3d2e1f0;
    

    hash32[0] = reverse(hash32[0]);
    hash32[1] = reverse(hash32[1]);
    hash32[2] = reverse(hash32[2]);
    hash32[3] = reverse(hash32[3]);
    hash32[4] = reverse(hash32[4]);
    hash32[5] = reverse(hash32[5]);
    hash32[6] = reverse(hash32[6]);
    hash32[7] = reverse(hash32[7]);

    return processedHash;
}

void MFNHashTypePlainCUDA_SHA256::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_SHA256::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceCharsetPlainSHA256",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceReverseCharsetPlainSHA256",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("charsetLengthsPlainSHA256",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("constantBitmapAPlainSHA256",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("passwordLengthPlainSHA256",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("numberOfHashesPlainSHA256",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalHashlistAddressPlainSHA256",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalBitmapAPlainSHA256",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalBitmapBPlainSHA256",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalBitmapCPlainSHA256",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalBitmapDPlainSHA256",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalBitmap256kPlainSHA256",
            &this->DeviceBitmap256kb_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalFoundPasswordsPlainSHA256",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainSHA256",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalStartPointsPlainSHA256",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceGlobalStartPasswords32PlainSHA256",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("deviceNumberThreadsPlainSHA256",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_SHA256_CopyValueToConstant("constantBitmapAPlainSHA256", 
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}