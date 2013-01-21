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

#include "MFN_CUDA_host/MFNHashTypeSaltedCUDA_MD5_PS.h"
#include "MFN_Common/MFNDebugging.h"

// NOTE: This constructor has to call all the way down!  I don't know what happens
// if you use different sizes for different classes - please don't do it!
MFNHashTypeSaltedCUDA_MD5_PS::MFNHashTypeSaltedCUDA_MD5_PS() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypeSaltedCUDA_MD5_PS::MFNHashTypeSaltedCUDA_MD5_PS()\n");
    
    this->hashAttributes.hashWordWidth32 = 1;
    this->hashAttributes.hashUsesSalt = 1;
    
}

void MFNHashTypeSaltedCUDA_MD5_PS::launchKernel() {
    trace_printf("MFNHashTypeSaltedCUDA_MD5_PS::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceNumberStepsToRunPlainMD5_PS",
        &this->perStep, sizeof(uint32_t));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceStartingSaltOffsetMD5_PS",
        &this->saltStartOffset, sizeof(uint32_t));

    MFNHashTypeSaltedCUDA_MD5_PS_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
}

void MFNHashTypeSaltedCUDA_MD5_PS::printLaunchDebugData() {
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

std::vector<uint8_t> MFNHashTypeSaltedCUDA_MD5_PS::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypeSaltedCUDA_MD5_PS::preProcessHash()\n");
  
    return rawHash;
}

std::vector<uint8_t> MFNHashTypeSaltedCUDA_MD5_PS::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypeSaltedCUDA_MD5_PS::postProcessHash()\n");
    
    
    return processedHash;
}

void MFNHashTypeSaltedCUDA_MD5_PS::copyConstantDataToDevice() {
    trace_printf("MFNHashTypeSaltedCUDA_MD5_PS::copyConstantDataToDevice()\n");
    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceCharsetPlainMD5_PS",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceReverseCharsetPlainMD5_PS",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("charsetLengthsPlainMD5_PS",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("constantBitmapAPlainMD5_PS",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("passwordLengthPlainMD5_PS",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("numberOfHashesPlainMD5_PS",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalHashlistAddressPlainMD5_PS",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalBitmapAPlainMD5_PS",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalBitmapBPlainMD5_PS",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalBitmapCPlainMD5_PS",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalBitmapDPlainMD5_PS",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalBitmap256kPlainMD5_PS",
            &this->DeviceBitmap256kb_Address, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalFoundPasswordsPlainMD5_PS",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainMD5_PS",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalStartPointsPlainMD5_PS",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalStartPasswords32PlainMD5_PS",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceNumberThreadsPlainMD5_PS",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("constantBitmapAPlainMD5_PS",
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    this->copySaltConstantsToDevice();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}

void MFNHashTypeSaltedCUDA_MD5_PS::copySaltConstantsToDevice() {
    trace_printf("MFNHashTypeSaltedCUDA_MD5_PS::copySaltConstantsToDevice()\n");
    cudaError_t err;
    // Salted hash data
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalSaltLengthsMD5_PS",
            &this->DeviceSaltLengthsAddress, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceGlobalSaltValuesMD5_PS",
            &this->DeviceSaltValuesAddress, sizeof(uint8_t *));
    MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant("deviceNumberOfSaltValues",
            &this->numberSaltsCopiedToDevice, sizeof(this->numberSaltsCopiedToDevice));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}
