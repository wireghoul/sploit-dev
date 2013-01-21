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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_DupMD5.h"
#include "MFN_Common/MFNDebugging.h"

MFNHashTypePlainCUDA_DupMD5::MFNHashTypePlainCUDA_DupMD5() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypePlainCUDA_DupMD5::MFNHashTypePlainCUDA_DupMD5()\n");
}

void MFNHashTypePlainCUDA_DupMD5::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_DupMD5::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceNumberStepsToRunPlainDupMD5",
        &this->perStep, sizeof(uint32_t));

    MFNHashTypePlainCUDA_DupMD5_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

std::vector<uint8_t> MFNHashTypePlainCUDA_DupMD5::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_DupMD5::preProcessHash()\n");
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

std::vector<uint8_t> MFNHashTypePlainCUDA_DupMD5::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_DupMD5::postProcessHash()\n");
    
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

void MFNHashTypePlainCUDA_DupMD5::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_DupMD5::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceCharsetPlainDupMD5",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceReverseCharsetPlainDupMD5",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("charsetLengthsPlainDupMD5",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("constantBitmapAPlainDupMD5",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("passwordLengthPlainDupMD5",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("numberOfHashesPlainDupMD5",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalHashlistAddressPlainDupMD5",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalBitmapAPlainDupMD5",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalBitmapBPlainDupMD5",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalBitmapCPlainDupMD5",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalBitmapDPlainDupMD5",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));
    
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalFoundPasswordsPlainDupMD5",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainDupMD5",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalStartPointsPlainDupMD5",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceGlobalStartPasswords32PlainDupMD5",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("deviceNumberThreadsPlainDupMD5",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant("constantBitmapAPlainDupMD5", 
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}