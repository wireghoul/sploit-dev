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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_DupNTLM.h"
#include "MFN_Common/MFNDebugging.h"

MFNHashTypePlainCUDA_DupNTLM::MFNHashTypePlainCUDA_DupNTLM() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypePlainCUDA_DupNTLM::MFNHashTypePlainCUDA_DupNTLM()\n");
}

void MFNHashTypePlainCUDA_DupNTLM::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_DupNTLM::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceNumberStepsToRunPlainDupNTLM",
        &this->perStep, sizeof(uint32_t));

    MFNHashTypePlainCUDA_DupNTLM_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

std::vector<uint8_t> MFNHashTypePlainCUDA_DupNTLM::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_DupNTLM::preProcessHash()\n");
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

std::vector<uint8_t> MFNHashTypePlainCUDA_DupNTLM::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_DupNTLM::postProcessHash()\n");
    
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

void MFNHashTypePlainCUDA_DupNTLM::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_DupNTLM::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceCharsetPlainDupNTLM",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceReverseCharsetPlainDupNTLM",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("charsetLengthsPlainDupNTLM",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("constantBitmapAPlainDupNTLM",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("passwordLengthPlainDupNTLM",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("numberOfHashesPlainDupNTLM",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalHashlistAddressPlainDupNTLM",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalBitmapAPlainDupNTLM",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalBitmapBPlainDupNTLM",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalBitmapCPlainDupNTLM",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalBitmapDPlainDupNTLM",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));
    
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalFoundPasswordsPlainDupNTLM",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainDupNTLM",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalStartPointsPlainDupNTLM",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceGlobalStartPasswords32PlainDupNTLM",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("deviceNumberThreadsPlainDupNTLM",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_DupNTLM_CopyValueToConstant("constantBitmapAPlainDupNTLM", 
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}