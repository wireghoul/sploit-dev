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
 * This file implements LM multihash cracking.
 */

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

#include "MFN_CUDA_device/MFN_CUDA_Common.h"
#include "MFN_CUDA_device/MFN_CUDA_DES.h"
#include "MFN_CUDA_device/MFN_CUDA_incrementors.h"

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
#define MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_PASSLEN 7

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainLM[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainLM[MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainLM[MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainLM[MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainLM;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainLM;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainLM;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainLM;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainLM;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainLM;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainLM;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainLM;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainLM;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainLM;
__device__ __constant__ uint64_t deviceNumberThreadsPlainLM;

__device__ inline void checkHash128LE_LM(uint32_t &a, uint32_t &b,
        uint32_t &b0, uint32_t &b1, uint8_t *sharedBitmapA,
        uint8_t *deviceGlobalBitmapA, uint8_t *deviceGlobalBitmapB,
        uint8_t *deviceGlobalFoundPasswords, uint8_t *deviceGlobalFoundPasswordFlags,
        uint8_t *deviceGlobalHashlistAddress, uint64_t numberOfHashes,
        uint8_t passwordLength) {
    if ((sharedBitmapA[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
        if (!(deviceGlobalBitmapA) || ((deviceGlobalBitmapA[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) {
            if (!deviceGlobalBitmapB || ((deviceGlobalBitmapB[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) {
                uint32_t search_high, search_low, search_index, current_hash_value;
                uint32_t *DEVICE_Hashes_32 = (uint32_t *) deviceGlobalHashlistAddress;
                search_high = numberOfHashes;
                search_low = 0;
                while (search_low < search_high) {
                    // Midpoint between search_high and search_low
                    search_index = search_low + (search_high - search_low) / 2;
                    current_hash_value = DEVICE_Hashes_32[2 * search_index];
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

                while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 2])) {
                    search_index--;
                }
                while ((a == DEVICE_Hashes_32[search_index * 2])) {
                    if (b == DEVICE_Hashes_32[search_index * 2 + 1]) {
                        // Copy the password to the correct location.
                        switch (passwordLength) {
                            case 7:
                                deviceGlobalFoundPasswords[search_index * passwordLength + 6] = (b1 >> 16) & 0xff;
                            case 6:
                                deviceGlobalFoundPasswords[search_index * passwordLength + 5] = (b1 >> 8) & 0xff;
                            case 5:
                                deviceGlobalFoundPasswords[search_index * passwordLength + 4] = (b1 >> 0) & 0xff;
                            case 4:
                                deviceGlobalFoundPasswords[search_index * passwordLength + 3] = (b0 >> 24) & 0xff;
                            case 3:
                                deviceGlobalFoundPasswords[search_index * passwordLength + 2] = (b0 >> 16) & 0xff;
                            case 2:
                                deviceGlobalFoundPasswords[search_index * passwordLength + 1] = (b0 >> 8) & 0xff;
                            case 1:
                                deviceGlobalFoundPasswords[search_index * passwordLength + 0] = (b0 >> 0) & 0xff;
                        }
                        deviceGlobalFoundPasswordFlags[search_index] = (unsigned char) MFN_PASSWORD_LM;
                    }
                    search_index++;
                }
            }
        }
    }
}

#define loadLMPasswords32(pa, dt, pl) { \
a = thread_index; \
b0 = pa[a]; \
if (pl > 3) {a += dt; b1 = pa[a];} \
ResetCharacterAtPosition(0, pl, b0, b1, b0, b0, b0, b0, b0, b0, b0, b0, b0, b0, b0, b0, b0, b0); \
}

#define storeLMPasswords32(pa, dt, pl) { \
pa[thread_index + 0] = b0; \
if (pl > 3) {pa[thread_index + (dt * 1)] = b1;} \
}


#define MAKE_MFN_LM_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_LM_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, a, b; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainLM[MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainLM[MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedCharsetLengthsPlainLM[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint32_t shared_des_skb[8][64]; \
    __shared__ uint32_t shared_des_SPtrans[8][64]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainLM; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainLM; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainLM; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainLM; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainLM; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_LM_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainLM[a] = charsetLengthsPlainLM[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
        for (a = 0; a < 8; a++) { \
            for (b = 0; b < 64; b++) { \
                shared_des_skb[a][b] = des_skb[a][b]; \
                shared_des_SPtrans[a][b] = des_SPtrans[a][b]; \
            } \
        } \
    } \
    syncthreads(); \
    b0 = b1 = 0; \
    loadLMPasswords32(deviceGlobalStartPasswords32PlainLM, deviceNumberThreadsPlainLM, pass_len); \
    while (password_count < deviceNumberStepsToRunPlainLM) { \
        cudaLM(b0, b1, a, b, shared_des_skb, shared_des_SPtrans); \
        checkHash128LE_LM(a, b, b0, b1, \
            sharedBitmap, \
            deviceGlobalBitmapAPlainLM, deviceGlobalBitmapBPlainLM, \
            deviceGlobalFoundPasswordsPlainLM, deviceGlobalFoundPasswordFlagsPlainLM, \
            deviceGlobalHashlistAddressPlainLM, numberOfHashesPlainLM, \
            passwordLengthPlainLM); \
        if (charsetLengthsPlainLM[1] == 0) { \
                makeMFNSingleIncrementors##pass_len (sharedCharsetPlainLM, sharedReverseCharsetPlainLM, sharedCharsetLengthsPlainLM); \
        } else { \
                makeMFNMultipleIncrementors##pass_len (sharedCharsetPlainLM, sharedReverseCharsetPlainLM, sharedCharsetLengthsPlainLM); \
        } \
        password_count++; \
    } \
    storeLMPasswords32(deviceGlobalStartPasswords32PlainLM, deviceNumberThreadsPlainLM, pass_len); \
}


MAKE_MFN_LM_KERNEL1_8LENGTH(1);
MAKE_MFN_LM_KERNEL1_8LENGTH(2);
MAKE_MFN_LM_KERNEL1_8LENGTH(3);
MAKE_MFN_LM_KERNEL1_8LENGTH(4);
MAKE_MFN_LM_KERNEL1_8LENGTH(5);
MAKE_MFN_LM_KERNEL1_8LENGTH(6);
MAKE_MFN_LM_KERNEL1_8LENGTH(7);

extern "C" cudaError_t MFNHashTypePlainCUDA_LM_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_LM_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_LM_LaunchKernel()\n");

//    cudaPrintfInit();
//    cudaError_t errbefore = cudaGetLastError();
//    if( cudaSuccess != errbefore)
//      {
//        printf("MFNHashTypePlainCUDA_LM Cuda errorbefore: %s.\n", cudaGetErrorString( errbefore) );
//      } else {
//        printf("No error before\n");
//      }


    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_LM_GeneratedKernel_1 <<< Blocks, Threads >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_LM_GeneratedKernel_2 <<< Blocks, Threads >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_LM_GeneratedKernel_3 <<< Blocks, Threads >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_LM_GeneratedKernel_4 <<< Blocks, Threads >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_LM_GeneratedKernel_5 <<< Blocks, Threads >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_LM_GeneratedKernel_6 <<< Blocks, Threads >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_LM_GeneratedKernel_7 <<< Blocks, Threads >>> ();
            break;
        default:
            printf("Password length %d unsupported!\n", passwordLength);
            exit(1);
            break;

    }
//    cudaPrintfDisplay(stdout, true);
//    cudaPrintfEnd();
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        printf("MFNHashTypePlainCUDA_LM Cuda error: %s.\n", cudaGetErrorString( err) );
      }

    return err;
}
