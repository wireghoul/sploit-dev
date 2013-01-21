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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_DoubleMD5.h"
#include "MFN_Common/MFNDebugging.h"

MFNHashTypePlainCUDA_DoubleMD5::MFNHashTypePlainCUDA_DoubleMD5() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypePlainCUDA_DoubleMD5::MFNHashTypePlainCUDA_DoubleMD5()\n");
}

void MFNHashTypePlainCUDA_DoubleMD5::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_DoubleMD5::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceNumberStepsToRunPlainDoubleMD5",
        &this->perStep, sizeof(uint32_t));

    //this->printLaunchDebugData(threadData);
    
    MFNHashTypePlainCUDA_DoubleMD5_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

void MFNHashTypePlainCUDA_DoubleMD5::printLaunchDebugData() {
    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);

    printf("Host value passwordLengthPlainDoubleMD5: %d\n", this->passwordLength);
    printf("Host value numberOfHashesPlainDoubleMD5: %lu\n", this->activeHashesProcessed.size());
    printf("Host value deviceGlobalHashlistAddressPlainDoubleMD5: 0x%16x\n", this->DeviceHashlistAddress);
    printf("Host value deviceGlobalBitmapAPlainDoubleMD5: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
    printf("Host value deviceGlobalBitmapBPlainDoubleMD5: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
    printf("Host value deviceGlobalBitmapCPlainDoubleMD5: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
    printf("Host value deviceGlobalBitmapDPlainDoubleMD5: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
    printf("Host value deviceGlobalFoundPasswordsPlainDoubleMD5: 0x%16x\n", this->DeviceFoundPasswordsAddress);
    printf("Host value deviceGlobalFoundPasswordFlagsPlainDoubleMD5: 0x%16x\n", this->DeviceSuccessAddress);
    printf("Host value deviceGlobalStartPointsPlainDoubleMD5: 0x%16x\n", this->DeviceStartPointAddress);
}

std::vector<uint8_t> MFNHashTypePlainCUDA_DoubleMD5::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_DoubleMD5::preProcessHash()\n");
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&rawHash[0];

    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];
    
    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;

    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainCUDA_DoubleMD5::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_DoubleMD5::postProcessHash()\n");
    
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&processedHash[0];
    
    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];
    
    a += 0x67452301;
    b += 0xefcdab89;
    c += 0x98badcfe;
    d += 0x10325476;
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;

    
    return processedHash;
}

void MFNHashTypePlainCUDA_DoubleMD5::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_DoubleMD5::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceCharsetPlainDoubleMD5",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceReverseCharsetPlainDoubleMD5",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("charsetLengthsPlainDoubleMD5",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("constantBitmapAPlainDoubleMD5",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("passwordLengthPlainDoubleMD5",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("numberOfHashesPlainDoubleMD5",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalHashlistAddressPlainDoubleMD5",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalBitmapAPlainDoubleMD5",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalBitmapBPlainDoubleMD5",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalBitmapCPlainDoubleMD5",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalBitmapDPlainDoubleMD5",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));
    
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalFoundPasswordsPlainDoubleMD5",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainDoubleMD5",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalStartPointsPlainDoubleMD5",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceGlobalStartPasswords32PlainDoubleMD5",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("deviceNumberThreadsPlainDoubleMD5",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant("constantBitmapAPlainDoubleMD5", 
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}