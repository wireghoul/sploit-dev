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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_16HEX.h"
#include "MFN_Common/MFNDebugging.h"


MFNHashTypePlainCUDA_16HEX::MFNHashTypePlainCUDA_16HEX() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypePlainCUDA_16HEX::MFNHashTypePlainCUDA_16HEX()\n");
}

void MFNHashTypePlainCUDA_16HEX::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_16HEX::launchKernel()\n");

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceNumberStepsToRunPlain16HEX",
        &this->perStep, sizeof(uint32_t));

    //this->printLaunchDebugData(threadData);
    
    MFNHashTypePlainCUDA_16HEX_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    
}

void MFNHashTypePlainCUDA_16HEX::printLaunchDebugData() {
    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);

    printf("Host value passwordLengthPlain16HEX: %d\n", this->passwordLength);
    printf("Host value numberOfHashesPlain16HEX: %lu\n", this->activeHashesProcessed.size());
    printf("Host value deviceGlobalHashlistAddressPlain16HEX: 0x%16x\n", this->DeviceHashlistAddress);
    printf("Host value deviceGlobalBitmapAPlain16HEX: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
    printf("Host value deviceGlobalBitmapBPlain16HEX: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
    printf("Host value deviceGlobalBitmapCPlain16HEX: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
    printf("Host value deviceGlobalBitmapDPlain16HEX: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
    printf("Host value deviceGlobalFoundPasswordsPlain16HEX: 0x%16x\n", this->DeviceFoundPasswordsAddress);
    printf("Host value deviceGlobalFoundPasswordFlagsPlain16HEX: 0x%16x\n", this->DeviceSuccessAddress);
    printf("Host value deviceGlobalStartPointsPlain16HEX: 0x%16x\n", this->DeviceStartPointAddress);
}


void MFNHashTypePlainCUDA_16HEX::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_16HEX::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceCharsetPlain16HEX",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceReverseCharsetPlain16HEX",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("charsetLengthsPlain16HEX",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("constantBitmapAPlain16HEX",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("passwordLengthPlain16HEX",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("numberOfHashesPlain16HEX",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalHashlistAddressPlain16HEX",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalBitmapAPlain16HEX",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalBitmapBPlain16HEX",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalBitmapCPlain16HEX",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalBitmapDPlain16HEX",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalBitmap256kPlain16HEX",
            &this->DeviceBitmap256kb_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalFoundPasswordsPlain16HEX",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlain16HEX",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalStartPointsPlain16HEX",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceGlobalStartPasswords32Plain16HEX",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("deviceNumberThreadsPlain16HEX",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_16HEX_CopyValueToConstant("constantBitmapAPlain16HEX", 
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}