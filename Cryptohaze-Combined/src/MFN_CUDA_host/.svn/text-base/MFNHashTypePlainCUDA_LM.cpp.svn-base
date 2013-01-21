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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_LM.h"
#include "MFN_Common/MFNDebugging.h"

MFNHashTypePlainCUDA_LM::MFNHashTypePlainCUDA_LM() :  MFNHashTypePlainCUDA(8) {
    trace_printf("MFNHashTypePlainCUDA_LM::MFNHashTypePlainCUDA_LM()\n");
}

void MFNHashTypePlainCUDA_LM::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_LM::launchKernel()\n");
    cudaError_t error;

    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceNumberStepsToRunPlainLM",
        &this->perStep, sizeof(uint32_t));
    error = cudaGetLastError();
    if( cudaSuccess != error)
      {
        printf("launchKernel Cuda error: %s.\n", cudaGetErrorString( error) );
      }
    //this->printLaunchDebugData(threadData);

    error = MFNHashTypePlainCUDA_LM_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    if (error != cudaSuccess) {
        printf("Thread %d: CUDA ERROR %s\n", this->threadId, cudaGetErrorString(error));
    }
}

void MFNHashTypePlainCUDA_LM::printLaunchDebugData() {
    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);

    printf("Host value passwordLengthPlainLM: %d\n", this->passwordLength);
    printf("Host value numberOfHashesPlainLM: %lu\n", this->activeHashesProcessed.size());
    printf("Host value deviceGlobalHashlistAddressPlainLM: 0x%16x\n", this->DeviceHashlistAddress);
    printf("Host value deviceGlobalBitmapAPlainLM: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
    printf("Host value deviceGlobalBitmapBPlainLM: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
    printf("Host value deviceGlobalBitmapCPlainLM: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
    printf("Host value deviceGlobalBitmapDPlainLM: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
    printf("Host value deviceGlobalFoundPasswordsPlainLM: 0x%16x\n", this->DeviceFoundPasswordsAddress);
    printf("Host value deviceGlobalFoundPasswordFlagsPlainLM: 0x%16x\n", this->DeviceSuccessAddress);
    printf("Host value deviceGlobalStartPointsPlainLM: 0x%16x\n", this->DeviceStartPointAddress);
}

std::vector<uint8_t> MFNHashTypePlainCUDA_LM::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_LM::preProcessHash()\n");
    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainCUDA_LM::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_LM::postProcessHash()\n");
    return processedHash;
}

void MFNHashTypePlainCUDA_LM::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_LM::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceCharsetPlainLM",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceReverseCharsetPlainLM",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("charsetLengthsPlainLM",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("constantBitmapAPlainLM",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("passwordLengthPlainLM",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("numberOfHashesPlainLM",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceGlobalHashlistAddressPlainLM",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceGlobalBitmapAPlainLM",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceGlobalBitmapBPlainLM",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceGlobalFoundPasswordsPlainLM",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainLM",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceGlobalStartPointsPlainLM",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceGlobalStartPasswords32PlainLM",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_LM_CopyValueToConstant("deviceNumberThreadsPlainLM",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_LM_CopyValueToConstant("constantBitmapAPlainLM",
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}