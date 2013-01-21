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
 * This file implements DoubleMD5 multihash cracking.
 */

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

//#include "CUDA_Common/cuPrintf.cu"

#include "MFN_CUDA_device/MFN_CUDA_incrementors.h"
#include "MFN_CUDA_device/MFN_CUDA_Common.h"
#include "MFN_CUDA_device/MFN_CUDA_MD5.h"

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
#define MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_PASSLEN 48

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainDoubleMD5[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainDoubleMD5[MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainDoubleMD5[MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainDoubleMD5[MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainDoubleMD5;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainDoubleMD5;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainDoubleMD5;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainDoubleMD5;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainDoubleMD5;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainDoubleMD5;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainDoubleMD5;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainDoubleMD5;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainDoubleMD5;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainDoubleMD5;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainDoubleMD5;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainDoubleMD5;
__device__ __constant__ uint64_t deviceNumberThreadsPlainDoubleMD5;





// Defined if we are using the loadPasswords32/storePasswords32
#define USE_NEW_PASSWORD_LOADING 1

__constant__ char hexLookupValuesDoubleMD5[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};


#define MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t b0pass, b1pass, b2pass, b3pass; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainDoubleMD5[MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainDoubleMD5[MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t               sharedCharsetLengthsPlainDoubleMD5[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint8_t               hashLookup[256][2]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainDoubleMD5; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainDoubleMD5; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainDoubleMD5; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainDoubleMD5; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainDoubleMD5; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainDoubleMD5[a] = charsetLengthsPlainDoubleMD5[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
        for (a = 0; a < 256; a++) { \
            hashLookup[a][0] = hexLookupValuesDoubleMD5[a / 16]; \
            hashLookup[a][1] = hexLookupValuesDoubleMD5[a % 16]; \
        } \
    } \
    syncthreads(); \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    b14 = pass_len * 8; \
    if (USE_NEW_PASSWORD_LOADING) { \
        loadPasswords32(deviceGlobalStartPasswords32PlainDoubleMD5, deviceNumberThreadsPlainDoubleMD5, pass_len); \
    } else {\
        if (charsetLengthsPlainDoubleMD5[1] == 0) { \
            loadPasswordSingle(sharedCharsetPlainDoubleMD5, deviceGlobalStartPointsPlainDoubleMD5, deviceNumberThreadsPlainDoubleMD5, pass_len); \
        } else { \
            loadPasswordMultiple(sharedCharsetPlainDoubleMD5, deviceGlobalStartPointsPlainDoubleMD5, deviceNumberThreadsPlainDoubleMD5, pass_len, MFN_HASH_TYPE_PLAIN_CUDA_DOUBLE_MD5_MAX_CHARSET_LENGTH); \
        } \
        ResetCharacterAtPosition(0x80, pass_len, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    } \
    while (password_count < deviceNumberStepsToRunPlainDoubleMD5) { \
        MD5_FULL_HASH(); \
        b0pass = b0; b1pass = b1; b2pass = b2; b3pass = b3; \
        LoadHash16AsLEString(hashLookup); \
        b8 = 0x00000080; \
        b14 = 32 * 8; \
        MD5_FIRST_3_ROUNDS(); \
        MD5II (a, b, c, d, b0, MD5S41, 0xf4292244); \
        MD5II (d, a, b, c, b7, MD5S42, 0x432aff97); \
        MD5II (c, d, a, b, b14, MD5S43, 0xab9423a7); \
        MD5II (b, c, d, a, b5, MD5S44, 0xfc93a039); \
        MD5II (a, b, c, d, b12, MD5S41, 0x655b59c3); \
        MD5II (d, a, b, c, b3, MD5S42, 0x8f0ccc92); \
        MD5II (c, d, a, b, b10, MD5S43, 0xffeff47d); \
        MD5II (b, c, d, a, b1, MD5S44, 0x85845dd1); \
        MD5II (a, b, c, d, b8, MD5S41, 0x6fa87e4f); \
        MD5II (d, a, b, c, b15, MD5S42, 0xfe2ce6e0); \
        MD5II (c, d, a, b, b6, MD5S43, 0xa3014314); \
        MD5II (b, c, d, a, b13, MD5S44, 0x4e0811a1); \
        MD5II (a, b, c, d, b4, MD5S41, 0xf7537e82);  \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            if (!(deviceGlobalBitmapAPlainDoubleMD5) || ((deviceGlobalBitmapAPlainDoubleMD5[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                MD5II (d, a, b, c, b11, MD5S42, 0xbd3af235); \
                if (!deviceGlobalBitmapDPlainDoubleMD5 || ((deviceGlobalBitmapDPlainDoubleMD5[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                    MD5II (c, d, a, b, b2, MD5S43, 0x2ad7d2bb);  \
                    if (!deviceGlobalBitmapCPlainDoubleMD5 || ((deviceGlobalBitmapCPlainDoubleMD5[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                        MD5II (b, c, d, a, b9, MD5S44, 0xeb86d391); \
                        if (!deviceGlobalBitmapBPlainDoubleMD5 || ((deviceGlobalBitmapBPlainDoubleMD5[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                            b0 = b0pass; b1 = b1pass; b2 = b2pass; b3 = b3pass; \
                            checkHashList128LE(a, b, c, d, b0, b1, b2, b3, \
                                b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
                                deviceGlobalFoundPasswordsPlainDoubleMD5, deviceGlobalFoundPasswordFlagsPlainDoubleMD5, \
                                deviceGlobalHashlistAddressPlainDoubleMD5, numberOfHashesPlainDoubleMD5, \
                                passwordLengthPlainDoubleMD5, MFN_PASSWORD_DOUBLE_MD5); \
        }   }   }   }   }\
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
        b14 = pass_len * 8; \
        b0 = b0pass; b1 = b1pass; b2 = b2pass; b3 = b3pass; \
        if (charsetLengthsPlainDoubleMD5[1] == 0) { \
                makeMFNSingleIncrementors##pass_len (sharedCharsetPlainDoubleMD5, sharedReverseCharsetPlainDoubleMD5, sharedCharsetLengthsPlainDoubleMD5); \
        } else { \
                makeMFNMultipleIncrementors##pass_len (sharedCharsetPlainDoubleMD5, sharedReverseCharsetPlainDoubleMD5, sharedCharsetLengthsPlainDoubleMD5); \
        } \
        password_count++; \
    } \
    if (USE_NEW_PASSWORD_LOADING) { \
        storePasswords32(deviceGlobalStartPasswords32PlainDoubleMD5, deviceNumberThreadsPlainDoubleMD5, pass_len); \
    } \
}

MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(1);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(2);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(3);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(4);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(5);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(6);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(7);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(8);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(9);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(10);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(11);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(12);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(13);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(14);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(15);
MAKE_MFN_DOUBLE_MD5_KERNEL1_8LENGTH(16);

extern "C" cudaError_t MFNHashTypePlainCUDA_DoubleMD5_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_DoubleMD5_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_MD5_LaunchKernel()\n");

    //cudaPrintfInit();
    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_1 <<< Blocks, Threads >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_2 <<< Blocks, Threads >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_3 <<< Blocks, Threads >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_4 <<< Blocks, Threads >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_5 <<< Blocks, Threads >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_6 <<< Blocks, Threads >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_7 <<< Blocks, Threads >>> ();
            break;
        case 8:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_8 <<< Blocks, Threads >>> ();
            break;
        case 9:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_9 <<< Blocks, Threads >>> ();
            break;
        case 10:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_10 <<< Blocks, Threads >>> ();
            break;
        case 11:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_11 <<< Blocks, Threads >>> ();
            break;
        case 12:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_12 <<< Blocks, Threads >>> ();
            break;
        case 13:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_13 <<< Blocks, Threads >>> ();
            break;
        case 14:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_14 <<< Blocks, Threads >>> ();
            break;
        case 15:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_15 <<< Blocks, Threads >>> ();
            break;
        case 16:
            MFNHashTypePlainCUDA_DoubleMD5_GeneratedKernel_16 <<< Blocks, Threads >>> ();
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
