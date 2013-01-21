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


/**
 * @section DESCRIPTION
 *
 * This file implements SHA1 multihash cracking.
 */

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

//#include "CUDA_Common/cuPrintf.cu"

#include "MFN_CUDA_device/MFN_CUDA_SHA_incrementors.h"
#include "MFN_CUDA_device/MFN_CUDA_Common.h"
#include "MFN_CUDA_device/MFN_CUDA_SHA1.h"

#if !defined(__CUDACC__)
    // define the keywords, so that the IDE does not complain about them
    #define __global__
    #define __device__
    #define __shared__
    #define __constant__
    #define blockIdx.x 1
    #define blockDim.x 1
    #define threadIdx.x 1
    #define __align__() /**/
#endif

/**
 * The maximum password length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_PASSLEN 48

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainSHA1[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainSHA1[MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainSHA1[MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainSHA1[MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainSHA1;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainSHA1;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainSHA1;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainSHA1;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainSHA1;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainSHA1;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainSHA1;
__device__ __constant__ uint8_t *deviceGlobalBitmap256kPlainSHA1;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainSHA1;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainSHA1;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainSHA1;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainSHA1;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainSHA1;
__device__ __constant__ uint64_t deviceNumberThreadsPlainSHA1;


// Defined if we are using the loadPasswords32/storePasswords32
#define USE_NEW_PASSWORD_LOADING 1

// Define SHA1 rotate left/right operators


__device__ inline void checkHashList160BE(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, 
        uint8_t *deviceGlobalFoundPasswords, uint8_t *deviceGlobalFoundPasswordFlags,
        uint8_t *deviceGlobalHashlistAddress, uint64_t numberOfHashes,
        uint8_t passwordLength) {
    uint32_t search_high, search_low, search_index, current_hash_value;
    uint32_t *DEVICE_Hashes_32 = (uint32_t *) deviceGlobalHashlistAddress;
    search_high = numberOfHashes;
    search_low = 0;
    while (search_low < search_high) {
        // Midpoint between search_high and search_low
        search_index = search_low + (search_high - search_low) / 2;
        current_hash_value = DEVICE_Hashes_32[5 * search_index];
        // Adjust search_high & search_low to work through space
        if (current_hash_value < a) {
            search_low = search_index + 1;
        } else {
            search_high = search_index;
        }
        if ((a == current_hash_value) && (search_low < numberOfHashes)) {
            // Break out of the search loop - search_index is on a value
            break;
        }
    }
    // Broke out of the while loop

    // If the loaded value does not match, there are no matches - just return.
    if (a != current_hash_value) {
        return;
    }
    // We've broken out of the loop, search_index should be on a matching value
    // Loop while the search index is the same - linear search through this to find all possible
    // matching passwords.
    // We first need to move backwards to the beginning, as we may be in the middle of a set of matching hashes.
    // If we are index 0, do NOT subtract, as we will wrap and this goes poorly.

    while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 5])) {
        search_index--;
    }
    while ((a == DEVICE_Hashes_32[search_index * 5])) {
        if (b == DEVICE_Hashes_32[search_index * 5 + 1]) {
            if (c == DEVICE_Hashes_32[search_index * 5 + 2]) {
                if (d == DEVICE_Hashes_32[search_index * 5 + 3]) {
                    // Copy the password to the correct location.
                    switch (passwordLength) {
                        case 16:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 15] = (b3 >> 0) & 0xff;
                        case 15:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 14] = (b3 >> 8) & 0xff;
                        case 14:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 13] = (b3 >> 16) & 0xff;
                        case 13:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 12] = (b3 >> 24) & 0xff;
                        case 12:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 11] = (b2 >> 0) & 0xff;
                        case 11:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 10] = (b2 >> 8) & 0xff;
                        case 10:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 9] = (b2 >> 16) & 0xff;
                        case 9:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 8] = (b2 >> 24) & 0xff;
                        case 8:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 7] = (b1 >> 0) & 0xff;
                        case 7:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 6] = (b1 >> 8) & 0xff;
                        case 6:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 5] = (b1 >> 16) & 0xff;
                        case 5:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 4] = (b1 >> 24) & 0xff;
                        case 4:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 3] = (b0 >> 0) & 0xff;
                        case 3:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 2] = (b0 >> 8) & 0xff;
                        case 2:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 1] = (b0 >> 16) & 0xff;
                        case 1:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 0] = (b0 >> 24) & 0xff;
                    }
                    deviceGlobalFoundPasswordFlags[search_index] = (unsigned char) MFN_PASSWORD_SHA1;
                }
            }
        }
        search_index++;
    }
}

// Extern storage for the plains.
extern __shared__ uint32_t plainStorageSHA1[]; \

#define MAKE_MFN_SHA1_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_SHA1_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, e; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainSHA1[MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainSHA1[MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t sharedCharsetLengthsPlainSHA1[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainSHA1; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainSHA1; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainSHA1; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainSHA1; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainSHA1; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainSHA1[a] = charsetLengthsPlainSHA1[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
    } \
    syncthreads(); \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    b15 = pass_len * 8; \
    if (USE_NEW_PASSWORD_LOADING) { \
        loadPasswords32(deviceGlobalStartPasswords32PlainSHA1, deviceNumberThreadsPlainSHA1, pass_len); \
    } else {\
        if (charsetLengthsPlainSHA1[1] == 0) { \
            loadPasswordSingle(sharedCharsetPlainSHA1, deviceGlobalStartPointsPlainSHA1, deviceNumberThreadsPlainSHA1, pass_len); \
        } else { \
            loadPasswordMultiple(sharedCharsetPlainSHA1, deviceGlobalStartPointsPlainSHA1, deviceNumberThreadsPlainSHA1, pass_len, MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH); \
        } \
        ResetCharacterAtPosition(0x80, pass_len, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    } \
    while (password_count < deviceNumberStepsToRunPlainSHA1) { \
        plainStorageSHA1[threadIdx.x] = b0; \
        if (pass_len > 3) {plainStorageSHA1[threadIdx.x + blockDim.x] = b1;} \
        if (pass_len > 7) {plainStorageSHA1[threadIdx.x + 2*blockDim.x] = b2;} \
        if (pass_len > 11) {plainStorageSHA1[threadIdx.x + 3*blockDim.x] = b3;} \
        if (pass_len > 15) {plainStorageSHA1[threadIdx.x + 4*blockDim.x] = b4;} \
        SHA1_PARTIAL_ROUNDS(); \
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
        b15 = pass_len * 8; \
        b0 = plainStorageSHA1[threadIdx.x]; \
        if (pass_len > 3) {b1 = plainStorageSHA1[threadIdx.x + blockDim.x];} \
        if (pass_len > 7) {b2 = plainStorageSHA1[threadIdx.x + 2*blockDim.x];} \
        if (pass_len > 11) {b3 = plainStorageSHA1[threadIdx.x + 3*blockDim.x];} \
        if (pass_len > 15) {b4 = plainStorageSHA1[threadIdx.x + 4*blockDim.x];} \
            if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
                if (!(deviceGlobalBitmap256kPlainSHA1) || ((deviceGlobalBitmap256kPlainSHA1[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!(deviceGlobalBitmapAPlainSHA1) || ((deviceGlobalBitmapAPlainSHA1[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapDPlainSHA1 || ((deviceGlobalBitmapDPlainSHA1[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapCPlainSHA1 || ((deviceGlobalBitmapCPlainSHA1[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                                if (!deviceGlobalBitmapBPlainSHA1 || ((deviceGlobalBitmapBPlainSHA1[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                    checkHashList160BE(a, b, c, d, b0, b1, b2, b3, \
                                        deviceGlobalFoundPasswordsPlainSHA1, deviceGlobalFoundPasswordFlagsPlainSHA1, \
                                        deviceGlobalHashlistAddressPlainSHA1, numberOfHashesPlainSHA1, \
                                        passwordLengthPlainSHA1); \
            }   }   }   }   }   } \
        if (charsetLengthsPlainSHA1[1] == 0) { \
                makeMFNSingleIncrementorsSHA##pass_len (sharedCharsetPlainSHA1, sharedReverseCharsetPlainSHA1, sharedCharsetLengthsPlainSHA1); \
        } else { \
                makeMFNMultipleIncrementorsSHA##pass_len (sharedCharsetPlainSHA1, sharedReverseCharsetPlainSHA1, sharedCharsetLengthsPlainSHA1); \
        } \
        password_count++; \
    } \
    if (USE_NEW_PASSWORD_LOADING) { \
        storePasswords32(deviceGlobalStartPasswords32PlainSHA1, deviceNumberThreadsPlainSHA1, pass_len); \
    } \
}

MAKE_MFN_SHA1_KERNEL1_8LENGTH(1);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(2);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(3);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(4);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(5);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(6);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(7);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(8);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(9);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(10);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(11);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(12);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(13);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(14);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(15);
MAKE_MFN_SHA1_KERNEL1_8LENGTH(16);

extern "C" cudaError_t MFNHashTypePlainCUDA_SHA1_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_SHA1_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_SHA1_LaunchKernel()\n");

    //cudaPrintfInit();
    
    // Calculate the amount of shared memory needed for the SHA1 kernels.
    // This is used to store the passwords between operations.
    int sharedMemoryBytesRequired = (((passwordLength + 1) / 4) + 1) * 4 * Threads;
    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_1 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_2 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_3 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_4 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_5 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_6 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_7 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 8:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_8 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 9:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_9 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 10:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_10 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 11:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_11 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 12:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_12 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 13:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_13 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 14:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_14 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 15:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_15 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 16:
            MFNHashTypePlainCUDA_SHA1_GeneratedKernel_16 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        default:
            printf("Password length %d unsupported!\n", passwordLength);
            exit(1);
            break;
            
    }
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();

    return cudaGetLastError();
}
