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

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_NTLM.h"
#include "MFN_Common/MFNDebugging.h"

#define MD4ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define MD4ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

#define MD4H(x, y, z) ((x) ^ (y) ^ (z))

#define MD4HH(a, b, c, d, x, s) { \
    (a) += MD4H ((b), (c), (d)) + (x) + (uint32_t)0x6ed9eba1; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }

#define REV_HH(a,b,c,d,data,shift) \
    a = MD4ROTATE_RIGHT((a), shift) - data - (uint32_t)0x6ed9eba1 - (b ^ c ^ d);



#define MD4S31 3
#define MD4S32 9
#define MD4S33 11
#define MD4S34 15


MFNHashTypePlainCUDA_NTLM::MFNHashTypePlainCUDA_NTLM() :  MFNHashTypePlainCUDA(16) {
    trace_printf("MFNHashTypePlainCUDA_NTLM::MFNHashTypePlainCUDA_NTLM()\n");
}

void MFNHashTypePlainCUDA_NTLM::launchKernel() {
    trace_printf("MFNHashTypePlainCUDA_NTLM::launchKernel()\n");
    cudaError_t error;
    
    // Copy the per-step data to the device.
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceNumberStepsToRunPlainNTLM",
        &this->perStep, sizeof(uint32_t));
    error = cudaGetLastError();
    if( cudaSuccess != error)
      {
        printf("launchKernel Cuda error: %s.\n", cudaGetErrorString( error) );
      }
    //this->printLaunchDebugData(threadData);

    error = MFNHashTypePlainCUDA_NTLM_LaunchKernel(this->passwordLength, this->GPUBlocks, this->GPUThreads);
    if (error != cudaSuccess) {
        printf("Thread %d: CUDA ERROR %s\n", this->threadId, cudaGetErrorString(error));
    }
}

void MFNHashTypePlainCUDA_NTLM::printLaunchDebugData() {
    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);

    printf("Host value passwordLengthPlainNTLM: %d\n", this->passwordLength);
    printf("Host value numberOfHashesPlainNTLM: %lu\n", this->activeHashesProcessed.size());
    printf("Host value deviceGlobalHashlistAddressPlainNTLM: 0x%16x\n", this->DeviceHashlistAddress);
    printf("Host value deviceGlobalBitmapAPlainNTLM: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
    printf("Host value deviceGlobalBitmapBPlainNTLM: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
    printf("Host value deviceGlobalBitmapCPlainNTLM: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
    printf("Host value deviceGlobalBitmapDPlainNTLM: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
    printf("Host value deviceGlobalFoundPasswordsPlainNTLM: 0x%16x\n", this->DeviceFoundPasswordsAddress);
    printf("Host value deviceGlobalFoundPasswordFlagsPlainNTLM: 0x%16x\n", this->DeviceSuccessAddress);
    printf("Host value deviceGlobalStartPointsPlainNTLM: 0x%16x\n", this->DeviceStartPointAddress);
}

std::vector<uint8_t> MFNHashTypePlainCUDA_NTLM::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCUDA_NTLM::preProcessHash()\n");
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&rawHash[0];

    /*
    printf("MFNHashTypePlainCUDA_NTLM::preProcessHash()\n");
    printf("Raw Hash: ");
    for (i = 0; i < rawHash.size(); i++) {
        printf("%02x", rawHash[i]);
    }
    printf("\n");
    */
    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];

    
    /**
     * Forward ops at the end past b1.
     * 
     * For length 6, all of this can be unwound.  b3 is known
     * as a padding bit.
     * 
     * For lengths >6, we can unwind back to b3.
    MD4HH (d, a, b, c, b9, MD4S32);
    MD4HH (c, d, a, b, b5, MD4S33);
    MD4HH (b, c, d, a, b13, MD4S34);
    MD4HH (a, b, c, d, b3, MD4S31);
    MD4HH (d, a, b, c, b11, MD4S32);
    MD4HH (c, d, a, b, b7, MD4S33);
    */
    
    // Always unwind the final constants
    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;

    // Always unwinding b15 - length field, always 0x00
    REV_HH(b, c, d, a, 0x00, MD4S34);

    if (this->passwordLength < 6) {
        // Unwind back through b9, with b3 = 0x00
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
        REV_HH (a, b, c, d, 0x00, MD4S31);
        REV_HH (b, c, d, a, 0x00, MD4S34);
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
    } else if (this->passwordLength == 6) {
        // Unwind through b9, with b3 = 0x00000080
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
        REV_HH (a, b, c, d, 0x80, MD4S31);
        REV_HH (b, c, d, a, 0x00, MD4S34);
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
    } else if (this->passwordLength < 14) {
        // Rewind through b3 with b7 = 0x00
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
    } else if (this->passwordLength == 14) {
        // Rewind through b3 with b7 = 0x80
        REV_HH (c, d, a, b, 0x80, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
    }
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;
    /*
    printf("Preprocessed Hash: ");
    for (i = 0; i < rawHash.size(); i++) {
        printf("%02x", rawHash[i]);
    }
    printf("\n");

    printf("Returning rawHash\n");
    */
    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainCUDA_NTLM::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCUDA_NTLM::postProcessHash()\n");

    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&processedHash[0];

    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];
    
    if (this->passwordLength < 6) {
        // Rewind back through b9, with b3 = 0x00
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
        MD4HH (b, c, d, a, 0x00, MD4S34);
        MD4HH (a, b, c, d, 0x00, MD4S31);
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
    } else if (this->passwordLength == 6) {
        // Rewind with b3 = 0x80
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
        MD4HH (b, c, d, a, 0x00, MD4S34);
        MD4HH (a, b, c, d, 0x80, MD4S31);
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
    } else if (this->passwordLength < 14) {
        // Rewind through b3 with b7 = 0x00
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
    } else if (this->passwordLength == 14) {
        // Rewind through b3 with b7 = 0x80
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x80, MD4S33);
    }

    // Always add b15 - will always be 0 (length field)
    MD4HH (b, c, d, a, 0x00, MD4S34);
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

void MFNHashTypePlainCUDA_NTLM::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA_NTLM::copyConstantDataToDevice()\n");

    cudaError_t err;

    // Begin copying constant data to the device.

    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceCharsetPlainNTLM",
            &this->charsetForwardLookup[0], this->charsetForwardLookup.size());
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceReverseCharsetPlainNTLM",
            &this->charsetReverseLookup[0], this->charsetReverseLookup.size());
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("charsetLengthsPlainNTLM",
            &this->charsetLengths[0], this->charsetLengths.size());
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("constantBitmapAPlainNTLM",
            &this->sharedBitmap8kb_a[0], 8192);

    uint8_t localPasswordLength = (uint8_t) this->passwordLength;
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("passwordLengthPlainNTLM",
            &localPasswordLength, sizeof(uint8_t));

    uint64_t localNumberHashes = (uint64_t) this->activeHashesProcessed.size();
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("numberOfHashesPlainNTLM",
            &localNumberHashes, sizeof(uint64_t));

    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalHashlistAddressPlainNTLM",
            &this->DeviceHashlistAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalBitmapAPlainNTLM",
            &this->DeviceBitmap128mb_a_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalBitmapBPlainNTLM",
            &this->DeviceBitmap128mb_b_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalBitmapCPlainNTLM",
            &this->DeviceBitmap128mb_c_Address, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalBitmapDPlainNTLM",
            &this->DeviceBitmap128mb_d_Address, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalFoundPasswordsPlainNTLM",
            &this->DeviceFoundPasswordsAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalFoundPasswordFlagsPlainNTLM",
            &this->DeviceSuccessAddress, sizeof(uint8_t *));

    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalStartPointsPlainNTLM",
            &this->DeviceStartPointAddress, sizeof(uint8_t *));
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceGlobalStartPasswords32PlainNTLM",
            &this->DeviceStartPasswords32Address, sizeof(uint8_t *));

    uint64_t localNumberThreads = this->GPUBlocks * this->GPUThreads;
    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("deviceNumberThreadsPlainNTLM",
            &localNumberThreads, sizeof(uint64_t));

    MFNHashTypePlainCUDA_NTLM_CopyValueToConstant("constantBitmapAPlainNTLM",
            &this->sharedBitmap8kb_a[0], this->sharedBitmap8kb_a.size());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d, dev %d: CUDA error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, cudaGetErrorString( err));
        exit(1);
    }
}