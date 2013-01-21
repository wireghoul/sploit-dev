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
 * This file implements 16HEX multihash cracking.
 */

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

//#include "CUDA_Common/cuPrintf.cu"

#include "MFN_CUDA_device/MFN_CUDA_Common.h"
#include "MFN_CUDA_device/MFN_CUDA_incrementors.h"
#include "MFN_CUDA_device/MFN_CUDA_MD5.h"
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
#define MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_PASSLEN 48

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlain16HEX[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlain16HEX[MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlain16HEX[MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlain16HEX[MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlain16HEX;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlain16HEX;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlain16HEX;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlain16HEX;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlain16HEX;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlain16HEX;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlain16HEX;
__device__ __constant__ uint8_t *deviceGlobalBitmap256kPlain16HEX;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlain16HEX;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlain16HEX;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlain16HEX;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32Plain16HEX;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlain16HEX;
__device__ __constant__ uint64_t deviceNumberThreadsPlain16HEX;



__constant__ char hexLookupValues16HEX[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};


extern __shared__ uint32_t plainStorage16HEX[];

#define MAKE_MFN_16HEX_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_16HEX_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlain16HEX[MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlain16HEX[MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t sharedCharsetLengthsPlain16HEX[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint8_t               hashLookup[256][2]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlain16HEX; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlain16HEX; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlain16HEX; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlain16HEX; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlain16HEX; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlain16HEX[a] = charsetLengthsPlain16HEX[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
        for (a = 0; a < 256; a++) { \
            hashLookup[a][0] = hexLookupValues16HEX[a / 16]; \
            hashLookup[a][1] = hexLookupValues16HEX[a % 16]; \
        } \
    } \
    syncthreads(); \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    loadPasswords32(deviceGlobalStartPasswords32Plain16HEX, deviceNumberThreadsPlain16HEX, pass_len); \
    while (password_count < deviceNumberStepsToRunPlain16HEX) { \
        /* Store the plains in the allocated space so they are available if an
         * algorithm such as SHA1 destroys them - or if we need to load them for
         * NTLM or something else like that. */ \
        StoreNormalPasswordInShared(plainStorage16HEX, pass_len); \
        b14 = pass_len * 8; \
        MD5_FULL_HASH(); \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            if (!(deviceGlobalBitmap256kPlain16HEX) || ((deviceGlobalBitmap256kPlain16HEX[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                if (!(deviceGlobalBitmapAPlain16HEX) || ((deviceGlobalBitmapAPlain16HEX[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!deviceGlobalBitmapDPlain16HEX || ((deviceGlobalBitmapDPlain16HEX[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapCPlain16HEX || ((deviceGlobalBitmapCPlain16HEX[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapBPlain16HEX || ((deviceGlobalBitmapBPlain16HEX[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LE(a, b, c, d, b0, b1, b2, b3, \
                                    b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
                                    deviceGlobalFoundPasswordsPlain16HEX, deviceGlobalFoundPasswordFlagsPlain16HEX, \
                                    deviceGlobalHashlistAddressPlain16HEX, numberOfHashesPlain16HEX, \
                                    passwordLengthPlain16HEX, MFN_PASSWORD_SINGLE_MD5); \
        }   }   }   }   }   } \
        LoadHash16AsLEString(hashLookup); \
        b8 = 0x00000080; \
        b14 = 32 * 8; \
        MD5_FULL_HASH(); \
        LoadNormalPasswordFromShared(plainStorage16HEX, pass_len); \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            if (!(deviceGlobalBitmap256kPlain16HEX) || ((deviceGlobalBitmap256kPlain16HEX[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                if (!(deviceGlobalBitmapAPlain16HEX) || ((deviceGlobalBitmapAPlain16HEX[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!deviceGlobalBitmapDPlain16HEX || ((deviceGlobalBitmapDPlain16HEX[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapCPlain16HEX || ((deviceGlobalBitmapCPlain16HEX[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapBPlain16HEX || ((deviceGlobalBitmapBPlain16HEX[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LE(a, b, c, d, b0, b1, b2, b3, \
                                    b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
                                    deviceGlobalFoundPasswordsPlain16HEX, deviceGlobalFoundPasswordFlagsPlain16HEX, \
                                    deviceGlobalHashlistAddressPlain16HEX, numberOfHashesPlain16HEX, \
                                    passwordLengthPlain16HEX, MFN_PASSWORD_DOUBLE_MD5); \
        }   }   }   }   }   } \
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
        LoadNormalPasswordFromShared(plainStorage16HEX, pass_len); \
        b14 = pass_len * 8; \
        /* MD4 uses the same length setup as MD5 - this is plain MD4, not NTLM. */ \
        MD4_FULL_HASH(); \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            if (!(deviceGlobalBitmap256kPlain16HEX) || ((deviceGlobalBitmap256kPlain16HEX[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                if (!(deviceGlobalBitmapAPlain16HEX) || ((deviceGlobalBitmapAPlain16HEX[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!deviceGlobalBitmapDPlain16HEX || ((deviceGlobalBitmapDPlain16HEX[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapCPlain16HEX || ((deviceGlobalBitmapCPlain16HEX[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapBPlain16HEX || ((deviceGlobalBitmapBPlain16HEX[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LE(a, b, c, d, b0, b1, b2, b3, \
                                    b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
                                    deviceGlobalFoundPasswordsPlain16HEX, deviceGlobalFoundPasswordFlagsPlain16HEX, \
                                    deviceGlobalHashlistAddressPlain16HEX, numberOfHashesPlain16HEX, \
                                    passwordLengthPlain16HEX, MFN_PASSWORD_MD4); \
        }   }   }   }   }   } \
        ExpandNTLMPasswordsFromShared(plainStorage16HEX, pass_len); \
        b14 = pass_len * 16; \
        MD4_FULL_HASH(); \
        checkHash128LENTLM(a, b, c, d, b0, b1, b2, b3, \
            b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
            sharedBitmap, \
            deviceGlobalBitmapAPlain16HEX, deviceGlobalBitmapBPlain16HEX, \
            deviceGlobalBitmapCPlain16HEX, deviceGlobalBitmapDPlain16HEX, \
            deviceGlobalFoundPasswordsPlain16HEX, deviceGlobalFoundPasswordFlagsPlain16HEX, \
            deviceGlobalHashlistAddressPlain16HEX, numberOfHashesPlain16HEX, \
            passwordLengthPlain16HEX); \
        /* Load the normal passwords back for the incrementors */ \
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
        LoadNormalPasswordFromShared(plainStorage16HEX, pass_len); \
        if (charsetLengthsPlain16HEX[1] == 0) { \
                makeMFNSingleIncrementors##pass_len (sharedCharsetPlain16HEX, sharedReverseCharsetPlain16HEX, sharedCharsetLengthsPlain16HEX); \
        } else { \
                makeMFNMultipleIncrementors##pass_len (sharedCharsetPlain16HEX, sharedReverseCharsetPlain16HEX, sharedCharsetLengthsPlain16HEX); \
        } \
        password_count++; \
    } \
    storePasswords32(deviceGlobalStartPasswords32Plain16HEX, deviceNumberThreadsPlain16HEX, pass_len); \
}


MAKE_MFN_16HEX_KERNEL1_8LENGTH(1);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(2);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(3);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(4);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(5);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(6);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(7);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(8);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(9);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(10);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(11);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(12);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(13);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(14);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(15);
MAKE_MFN_16HEX_KERNEL1_8LENGTH(16);

extern "C" cudaError_t MFNHashTypePlainCUDA_16HEX_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_16HEX_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_16HEX_LaunchKernel()\n");

    // Calculate the amount of shared memory needed for the SHA1 kernels.
    // This is used to store the passwords between operations.
    int sharedMemoryBytesRequired = (((passwordLength + 1) / 4) + 1) * 4 * Threads;

    //cudaPrintfInit();
    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_1 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_2 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_3 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_4 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_5 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_6 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_7 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 8:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_8 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 9:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_9 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 10:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_10 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 11:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_11 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 12:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_12 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 13:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_13 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 14:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_14 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 15:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_15 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 16:
            MFNHashTypePlainCUDA_16HEX_GeneratedKernel_16 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
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
