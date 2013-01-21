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
 * This file implements NTLM multihash cracking.
 */

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

#include "MFN_CUDA_device/MFN_CUDA_Common.h"
#include "MFN_CUDA_device/MFN_CUDA_NTLM_incrementors.h"
#include "MFN_CUDA_device/MFN_CUDA_MD4.h"

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
#define MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_PASSLEN 28

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainNTLM[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainNTLM[MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainNTLM[MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainNTLM[MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainNTLM;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainNTLM;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainNTLM;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainNTLM;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainNTLM;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainNTLM;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainNTLM;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainNTLM;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainNTLM;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainNTLM;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainNTLM;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainNTLM;
__device__ __constant__ uint64_t deviceNumberThreadsPlainNTLM;




#define MAKE_MFN_NTLM_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_NTLM_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainNTLM[MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainNTLM[MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedCharsetLengthsPlainNTLM[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainNTLM; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainNTLM; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainNTLM; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainNTLM; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainNTLM; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainNTLM[a] = charsetLengthsPlainNTLM[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
    } \
    syncthreads(); \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    b14 = pass_len * 16; \
    loadNTLMPasswords32(deviceGlobalStartPasswords32PlainNTLM, deviceNumberThreadsPlainNTLM, pass_len); \
    while (password_count < deviceNumberStepsToRunPlainNTLM) { \
        MD4_FIRST_2_ROUNDS(); \
        MD4HH (a, b, c, d, b0, MD4S31); \
        MD4HH (d, a, b, c, b8, MD4S32); \
        MD4HH (c, d, a, b, b4, MD4S33); \
        MD4HH (b, c, d, a, b12, MD4S34); \
        MD4HH (a, b, c, d, b2, MD4S31); \
        MD4HH (d, a, b, c, b10, MD4S32); \
        MD4HH (c, d, a, b, b6, MD4S33); \
        MD4HH (b, c, d, a, b14, MD4S34); \
        MD4HH (a, b, c, d, b1, MD4S31); \
        if (pass_len > 6) { \
            MD4HH (d, a, b, c, b9, MD4S32); \
            MD4HH (c, d, a, b, b5, MD4S33); \
            MD4HH (b, c, d, a, b13, MD4S34); \
            MD4HH (a, b, c, d, b3, MD4S31); \
            if (pass_len > 14) { \
                MD4HH (d, a, b, c, b11, MD4S32); \
                MD4HH (c, d, a, b, b7, MD4S33); \
            } \
        } \
        checkHash128LENTLM(a, b, c, d, b0, b1, b2, b3, \
            b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
            sharedBitmap, \
            deviceGlobalBitmapAPlainNTLM, deviceGlobalBitmapBPlainNTLM, \
            deviceGlobalBitmapCPlainNTLM, deviceGlobalBitmapDPlainNTLM, \
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM, \
            deviceGlobalHashlistAddressPlainNTLM, numberOfHashesPlainNTLM, \
            passwordLengthPlainNTLM); \
        if (charsetLengthsPlainNTLM[1] == 0) { \
                makeMFNSingleIncrementorsNTLM##pass_len (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM); \
        } else { \
                makeMFNMultipleIncrementorsNTLM##pass_len (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM); \
        } \
        password_count++; \
    } \
    storeNTLMPasswords32(deviceGlobalStartPasswords32PlainNTLM, deviceNumberThreadsPlainNTLM, pass_len); \
}


MAKE_MFN_NTLM_KERNEL1_8LENGTH(1);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(2);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(3);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(4);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(5);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(6);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(7);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(8);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(9);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(10);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(11);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(12);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(13);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(14);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(15);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(16);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(17);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(18);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(19);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(20);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(21);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(22);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(23);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(24);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(25);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(26);
MAKE_MFN_NTLM_KERNEL1_8LENGTH(27);

extern "C" cudaError_t MFNHashTypePlainCUDA_NTLM_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_NTLM_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_NTLM_LaunchKernel()\n");
    
    //cudaPrintfInit();
//    cudaError_t errbefore = cudaGetLastError();
//    if( cudaSuccess != errbefore)
//      {
//        printf("MFNHashTypePlainCUDA_NTLM Cuda errorbefore: %s.\n", cudaGetErrorString( errbefore) );
//      } else {
//        printf("No error before\n");
//      }

    
    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_1 <<< Blocks, Threads >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_2 <<< Blocks, Threads >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_3 <<< Blocks, Threads >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_4 <<< Blocks, Threads >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_5 <<< Blocks, Threads >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_6 <<< Blocks, Threads >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_7 <<< Blocks, Threads >>> ();
            break;
        case 8:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_8 <<< Blocks, Threads >>> ();
            break;
        case 9:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_9 <<< Blocks, Threads >>> ();
            break;
        case 10:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_10 <<< Blocks, Threads >>> ();
            break;
        case 11:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_11 <<< Blocks, Threads >>> ();
            break;
        case 12:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_12 <<< Blocks, Threads >>> ();
            break;
        case 13:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_13 <<< Blocks, Threads >>> ();
            break;
        case 14:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_14 <<< Blocks, Threads >>> ();
            break;
        case 15:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_15 <<< Blocks, Threads >>> ();
            break;
        case 16:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_16 <<< Blocks, Threads >>> ();
            break;
        case 17:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_17 <<< Blocks, Threads >>> ();
            break;
        case 18:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_18 <<< Blocks, Threads >>> ();
            break;
        case 19:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_19 <<< Blocks, Threads >>> ();
            break;
        case 20:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_20 <<< Blocks, Threads >>> ();
            break;
        case 21:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_21 <<< Blocks, Threads >>> ();
            break;
        case 22:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_22 <<< Blocks, Threads >>> ();
            break;
        case 23:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_23 <<< Blocks, Threads >>> ();
            break;
        case 24:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_24 <<< Blocks, Threads >>> ();
            break;
        case 25:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_25 <<< Blocks, Threads >>> ();
            break;
        case 26:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_26 <<< Blocks, Threads >>> ();
            break;
        case 27:
            MFNHashTypePlainCUDA_NTLM_GeneratedKernel_27 <<< Blocks, Threads >>> ();
            break;
        default:
            printf("Password length %d unsupported!\n", passwordLength);
            exit(1);
            break;

    }
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        printf("MFNHashTypePlainCUDA_NTLM Cuda error: %s.\n", cudaGetErrorString( err) );
      }

    return err;
}
