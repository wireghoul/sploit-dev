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
 * This file implements MD5 pass/salt multihash cracking.
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
#define MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_PASSLEN 48

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainMD5_PS[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainMD5_PS[MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainMD5_PS[MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainMD5_PS[MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainMD5_PS;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainMD5_PS;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainMD5_PS;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainMD5_PS;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainMD5_PS;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainMD5_PS;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainMD5_PS;
__device__ __constant__ uint8_t *deviceGlobalBitmap256kPlainMD5_PS;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainMD5_PS;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainMD5_PS;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainMD5_PS;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainMD5_PS;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainMD5_PS;
__device__ __constant__ uint64_t deviceNumberThreadsPlainMD5_PS;

// Salted hash data
__device__ __constant__ uint32_t *deviceGlobalSaltLengthsMD5_PS;
__device__ __constant__ uint32_t *deviceGlobalSaltValuesMD5_PS;
__device__ __constant__ uint32_t deviceNumberOfSaltValues;
__device__ __constant__ uint32_t deviceStartingSaltOffsetMD5_PS;

#define LOAD_SALTS(pass_len, salt_len, salt_index, salt_array, num_salts) { \
if (pass_len < 4) { \
    /*cuPrintf("1 Switch value: %d\n", ((pass_len + salt_len + 1) / 4) + 1);*/ \
    switch(((pass_len + salt_len) / 4) + 1) { \
        case 4: \
            b3 |= salt_array[salt_index + num_salts * 3]; \
        case 3: \
            b2 |= salt_array[salt_index + num_salts * 2]; \
        case 2: \
            b1 |= salt_array[salt_index + num_salts * 1]; \
        case 1: \
            b0 |= salt_array[salt_index + num_salts * 0]; \
    } \
  } \
else if (pass_len < 8) { \
    /*cuPrintf("2 Switch value: %d\n", ((pass_len + salt_len + 1) / 4) + 1);*/ \
    switch(((pass_len + salt_len) / 4) + 1) { \
        case 5: \
            b4 |= salt_array[salt_index + num_salts * 3]; \
        case 4: \
            b3 |= salt_array[salt_index + num_salts * 2]; \
        case 3: \
            b2 |= salt_array[salt_index + num_salts * 1]; \
        case 2: \
            b1 |= salt_array[salt_index + num_salts * 0]; \
    } \
  } \
else if (pass_len < 12) { \
    /*cuPrintf("2 Switch value: %d\n", ((pass_len + salt_len + 1) / 4) + 1);*/ \
    switch(((pass_len + salt_len) / 4) + 1) { \
        case 6: \
            b5 |= salt_array[salt_index + num_salts * 3]; \
        case 5: \
            b4 |= salt_array[salt_index + num_salts * 2]; \
        case 4: \
            b3 |= salt_array[salt_index + num_salts * 1]; \
        case 3: \
            b2 |= salt_array[salt_index + num_salts * 0]; \
    } \
  } \
}

#define CLEAR_SALT_REGION(pass_len, salt_len) { \
switch(pass_len) { \
    case 1: \
        b0 &= 0x000000ff; \
        break; \
    case 2: \
        b0 &= 0x0000ffff; \
        break; \
    case 3: \
        b0 &= 0x00ffffff; \
        break; \
    case 4: \
        break; \
    case 5: \
        b1 &= 0x000000ff; \
        break; \
    case 6: \
        b1 &= 0x0000ffff; \
        break; \
    case 7: \
        b1 &= 0x00ffffff; \
        break; \
    case 8: \
        break; \
    } \
if (pass_len <= 4) { \
    b1 = b2 = b3 = b4 = 0; \
} else if (pass_len <= 8) { \
    b2 = b3 = b4 = b5 = 0; \
}\
}


#define MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t password_count = 0, passOffset, saltIndex; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainMD5_PS[MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainMD5_PS[MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t sharedCharsetLengthsPlainMD5_PS[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainMD5_PS; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainMD5_PS; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainMD5_PS; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainMD5_PS; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainMD5_PS; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_MD5_PS_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainMD5_PS[a] = charsetLengthsPlainMD5_PS[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
    } \
    syncthreads(); \
    saltIndex = deviceStartingSaltOffsetMD5_PS; \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    loadPasswords32(deviceGlobalStartPasswords32PlainMD5_PS, deviceNumberThreadsPlainMD5_PS, pass_len); \
    while (password_count < deviceNumberStepsToRunPlainMD5_PS) { \
        /*cuPrintf("saltIndex: %u\n", saltIndex);*/ \
        CLEAR_SALT_REGION(pass_len, deviceGlobalSaltLengthsMD5_PS[saltIndex]); \
        /*cuPrintf("1 %08x %08x %08x %08x %08x ... %08x\n", b0, b1, b2, b3, b4, b14);*/ \
        LOAD_SALTS(pass_len, deviceGlobalSaltLengthsMD5_PS[saltIndex], \
            saltIndex, deviceGlobalSaltValuesMD5_PS, deviceNumberOfSaltValues); \
        b14 = (pass_len + deviceGlobalSaltLengthsMD5_PS[saltIndex]) * 8; \
        /*cuPrintf("2 %08x %08x %08x %08x %08x ... %08x\n", b0, b1, b2, b3, b4, b14);*/ \
        MD5_FULL_HASH(); \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            if (!(deviceGlobalBitmap256kPlainMD5_PS) || ((deviceGlobalBitmap256kPlainMD5_PS[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                if (!(deviceGlobalBitmapAPlainMD5_PS) || ((deviceGlobalBitmapAPlainMD5_PS[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!deviceGlobalBitmapDPlainMD5_PS || ((deviceGlobalBitmapDPlainMD5_PS[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapCPlainMD5_PS || ((deviceGlobalBitmapCPlainMD5_PS[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapBPlainMD5_PS || ((deviceGlobalBitmapBPlainMD5_PS[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LE(a, b, c, d, b0, b1, b2, b3, \
                                    b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
                                    deviceGlobalFoundPasswordsPlainMD5_PS, deviceGlobalFoundPasswordFlagsPlainMD5_PS, \
                                    deviceGlobalHashlistAddressPlainMD5_PS, numberOfHashesPlainMD5_PS, \
                                    passwordLengthPlainMD5_PS, MFN_PASSWORD_SINGLE_MD5); \
        }   }   }   }   }   } \
        saltIndex++; \
        if (saltIndex >= deviceNumberOfSaltValues) { \
        /*cuPrintf("Resetting salt index to 0.\n");*/ \
        saltIndex = 0; \
            if (charsetLengthsPlainMD5_PS[1] == 0) { \
                    makeMFNSingleIncrementors##pass_len (sharedCharsetPlainMD5_PS, sharedReverseCharsetPlainMD5_PS, sharedCharsetLengthsPlainMD5_PS); \
            } else { \
                    makeMFNMultipleIncrementors##pass_len (sharedCharsetPlainMD5_PS, sharedReverseCharsetPlainMD5_PS, sharedCharsetLengthsPlainMD5_PS); \
            } \
        } \
        password_count++; \
    } \
    /*cuPrintf("Quitting %08x %08x %08x %08x %08x ... %08x\n", b0, b1, b2, b3, b4, b14);*/ \
    storePasswords32(deviceGlobalStartPasswords32PlainMD5_PS, deviceNumberThreadsPlainMD5_PS, pass_len); \
}

MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(1);
MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(2);
MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(3);
MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(4);
MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(5);
MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(6);
MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(7);
MAKE_MFN_MD5_PS_KERNEL1_8LENGTH(8);

extern "C" cudaError_t MFNHashTypeSaltedCUDA_MD5_PS_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypeSaltedCUDA_MD5_PS_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypeSaltedCUDA_MD5_PS_LaunchKernel()\n");

    //cudaPrintfInit();
    switch (passwordLength) {
        case 1:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_1 <<< Blocks, Threads >>> ();
            break;
        case 2:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_2 <<< Blocks, Threads >>> ();
            break;
        case 3:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_3 <<< Blocks, Threads >>> ();
            break;
        case 4:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_4 <<< Blocks, Threads >>> ();
            break;
        case 5:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_5 <<< Blocks, Threads >>> ();
            break;
        case 6:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_6 <<< Blocks, Threads >>> ();
            break;
        case 7:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_7 <<< Blocks, Threads >>> ();
            break;
        case 8:
            MFNHashTypeSaltedCUDA_MD5_PS_GeneratedKernel_8 <<< Blocks, Threads >>> ();
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
