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
 * This file implements Lotus multihash cracking.
 */

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

//#include "CUDA_Common/cuPrintf.cu"

#include "MFN_CUDA_device/MFN_CUDA_incrementors.h"
#include "MFN_CUDA_device/MFN_CUDA_Common.h"

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

// For cards with 48k shared mem, use 512 threads.  Else use 64.
#if __CUDA_ARCH__ >= 200
#define SHARED_MEM_THREADS 512
#else
#define SHARED_MEM_THREADS 64
#endif

/**
 * The maximum password length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_PASSLEN 16

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__  uint8_t constantBitmapAPlainLOTUS[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainLOTUS[MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainLOTUS[MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainLOTUS[MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainLOTUS;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainLOTUS;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainLOTUS;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainLOTUS;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainLOTUS;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainLOTUS;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainLOTUS;
__device__ __constant__ uint8_t *deviceGlobalBitmap256kPlainLOTUS;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainLOTUS;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainLOTUS;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainLOTUS;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainLOTUS;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainLOTUS;
__device__ __constant__ uint64_t deviceNumberThreadsPlainLOTUS;

__device__ __constant__ uint8_t __align__(16) S_BOX[] = {
  0xBD,0x56,0xEA,0xF2,0xA2,0xF1,0xAC,0x2A,
  0xB0,0x93,0xD1,0x9C,0x1B,0x33,0xFD,0xD0,
  0x30,0x04,0xB6,0xDC,0x7D,0xDF,0x32,0x4B,
  0xF7,0xCB,0x45,0x9B,0x31,0xBB,0x21,0x5A,
  0x41,0x9F,0xE1,0xD9,0x4A,0x4D,0x9E,0xDA,
  0xA0,0x68,0x2C,0xC3,0x27,0x5F,0x80,0x36,
  0x3E,0xEE,0xFB,0x95,0x1A,0xFE,0xCE,0xA8,
  0x34,0xA9,0x13,0xF0,0xA6,0x3F,0xD8,0x0C,
  0x78,0x24,0xAF,0x23,0x52,0xC1,0x67,0x17,
  0xF5,0x66,0x90,0xE7,0xE8,0x07,0xB8,0x60,
  0x48,0xE6,0x1E,0x53,0xF3,0x92,0xA4,0x72,
  0x8C,0x08,0x15,0x6E,0x86,0x00,0x84,0xFA,
  0xF4,0x7F,0x8A,0x42,0x19,0xF6,0xDB,0xCD,
  0x14,0x8D,0x50,0x12,0xBA,0x3C,0x06,0x4E,
  0xEC,0xB3,0x35,0x11,0xA1,0x88,0x8E,0x2B,
  0x94,0x99,0xB7,0x71,0x74,0xD3,0xE4,0xBF,
  0x3A,0xDE,0x96,0x0E,0xBC,0x0A,0xED,0x77,
  0xFC,0x37,0x6B,0x03,0x79,0x89,0x62,0xC6,
  0xD7,0xC0,0xD2,0x7C,0x6A,0x8B,0x22,0xA3,
  0x5B,0x05,0x5D,0x02,0x75,0xD5,0x61,0xE3,
  0x18,0x8F,0x55,0x51,0xAD,0x1F,0x0B,0x5E,
  0x85,0xE5,0xC2,0x57,0x63,0xCA,0x3D,0x6C,
  0xB4,0xC5,0xCC,0x70,0xB2,0x91,0x59,0x0D,
  0x47,0x20,0xC8,0x4F,0x58,0xE0,0x01,0xE2,
  0x16,0x38,0xC4,0x6F,0x3B,0x0F,0x65,0x46,
  0xBE,0x7E,0x2D,0x7B,0x82,0xF9,0x40,0xB5,
  0x1D,0x73,0xF8,0xEB,0x26,0xC7,0x87,0x97,
  0x25,0x54,0xB1,0x28,0xAA,0x98,0x9D,0xA5,
  0x64,0x6D,0x7A,0xD4,0x10,0x81,0x44,0xEF,
  0x49,0xD6,0xAE,0x2E,0xDD,0x76,0x5C,0x2F,
  0xA7,0x1C,0xC9,0x09,0x69,0x9A,0x83,0xCF,
  0x29,0x39,0xB9,0xE9,0x4C,0xFF,0x43,0xAB
};

// To avoid shared conflicts, the work space is broken into banks of 4 (32-bits) wide to match shared mem.
// This gets the offset in the shared space based on the thread count and requested offset.
#define lotusSharedIndex(offset) (((offset) % 4) + (((offset) / 4) * blockDim.x * 4) + (threadIdx.x * 4))
#define lotusSharedIndex32(offset) ((offset) * blockDim.x + threadIdx.x)

#define MAKE_MFN_LOTUS_KERNEL1_16LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_##pass_len () { \
    const uint8_t inverseSize = (16 - pass_len); \
    int i, j; \
    uint8_t offset, X; \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, a, b, c, d; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainLOTUS[MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainLOTUS[MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t sharedCharsetLengthsPlainLOTUS[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint8_t __align__(16)  lotusWorkingSpace[SHARED_MEM_THREADS*64]; \
    uint32_t *lotusWorkingSpace32 = (uint32_t *)lotusWorkingSpace; \
    __device__ __shared__ uint8_t __align__(16) S_S_BOX[256]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainLOTUS; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainLOTUS; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainLOTUS; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainLOTUS; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainLOTUS; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        uint64_t *constantSbox64 = (uint64_t *)S_BOX; \
        uint64_t *sharedSbox64 = (uint64_t *)S_S_BOX; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainLOTUS[a] = charsetLengthsPlainLOTUS[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
        for (a = 0; a < (256 / 8); a++) { \
            sharedSbox64[a] = constantSbox64[a]; \
        } \
    } \
    syncthreads(); \
    loadPasswords32(deviceGlobalStartPasswords32PlainLOTUS, deviceNumberThreadsPlainLOTUS, pass_len); \
    while (password_count < deviceNumberStepsToRunPlainLOTUS) { \
        lotusWorkingSpace32[lotusSharedIndex32(0)] = 0xd3503c3e; \
        lotusWorkingSpace32[lotusSharedIndex32(1)] = 0x5cc587ab; \
        lotusWorkingSpace32[lotusSharedIndex32(2)] = 0x9db9d4bc; \
        lotusWorkingSpace32[lotusSharedIndex32(3)] = 0x29d76e38; \
        for (i = 16; i < 48; i++) { \
            lotusWorkingSpace[lotusSharedIndex(i)] = inverseSize; \
        } \
        uint32_t passMask = 0; \
        switch (pass_len % 4) { \
            case 0: \
                passMask = 0xffffffff; \
                break; \
            case 1: \
                passMask = 0x000000ff; \
                break; \
            case 2: \
                passMask = 0x0000ffff; \
                break; \
            case 3: \
                passMask = 0x00ffffff; \
                break; \
        } \
        if (pass_len <= 4) { \
            lotusWorkingSpace32[lotusSharedIndex32(4)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(4)] |= (b0 & passMask); \
            lotusWorkingSpace32[lotusSharedIndex32(8)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(8)] |= (b0 & passMask); \
        } else if (pass_len <= 8) { \
            lotusWorkingSpace32[lotusSharedIndex32(4)] = b0; \
            lotusWorkingSpace32[lotusSharedIndex32(5)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(5)] |= (b1 & passMask); \
            lotusWorkingSpace32[lotusSharedIndex32(8)] = b0; \
            lotusWorkingSpace32[lotusSharedIndex32(9)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(9)] |= (b1 & passMask); \
        } else if (pass_len <= 12) { \
            lotusWorkingSpace32[lotusSharedIndex32(4)] = b0; \
            lotusWorkingSpace32[lotusSharedIndex32(5)] = b1; \
            lotusWorkingSpace32[lotusSharedIndex32(6)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(6)] |= (b2 & passMask); \
            lotusWorkingSpace32[lotusSharedIndex32(8)] = b0; \
            lotusWorkingSpace32[lotusSharedIndex32(9)] = b1; \
            lotusWorkingSpace32[lotusSharedIndex32(10)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(10)] |= (b2 & passMask); \
        } else if (pass_len <= 16) { \
            lotusWorkingSpace32[lotusSharedIndex32(4)] = b0; \
            lotusWorkingSpace32[lotusSharedIndex32(5)] = b1; \
            lotusWorkingSpace32[lotusSharedIndex32(6)] = b2; \
            lotusWorkingSpace32[lotusSharedIndex32(7)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(7)] |= (b3 & passMask); \
            lotusWorkingSpace32[lotusSharedIndex32(8)] = b0; \
            lotusWorkingSpace32[lotusSharedIndex32(9)] = b1; \
            lotusWorkingSpace32[lotusSharedIndex32(10)] = b2; \
            lotusWorkingSpace32[lotusSharedIndex32(11)] &= ~passMask; \
            lotusWorkingSpace32[lotusSharedIndex32(11)] |= (b3 & passMask); \
        } \
        lotusWorkingSpace[lotusSharedIndex(48)] = S_S_BOX[lotusWorkingSpace[lotusSharedIndex(16)]]; \
        /* Generate the key based on the loaded password. */  \
        for(i = 1; i < 16; i ++) { \
            lotusWorkingSpace[lotusSharedIndex(48 + i)] =  \
                    S_S_BOX[(uint8_t)(lotusWorkingSpace[lotusSharedIndex(48 + i - 1)] ^  \
                    lotusWorkingSpace[lotusSharedIndex(16 + i)])]; \
        } \
        offset = 32; \
        X = lotusWorkingSpace[lotusSharedIndex(15)]; \
        for (i = 16; i < 48; i++, offset--) { \
            X = lotusWorkingSpace[lotusSharedIndex(i)] ^= S_S_BOX[(uint8_t)(offset + X)]; \
        } \
        for (i = 17; i > 0; i--) { \
            offset = 48; \
            for (j = 0; j < 48; j++, offset--) { \
                X = lotusWorkingSpace[lotusSharedIndex(j)] ^= S_S_BOX[(uint8_t)(offset + X)]; \
            } \
        } \
        lotusWorkingSpace32[lotusSharedIndex32(4)] = lotusWorkingSpace32[lotusSharedIndex32(12)]; \
        lotusWorkingSpace32[lotusSharedIndex32(5)] = lotusWorkingSpace32[lotusSharedIndex32(13)]; \
        lotusWorkingSpace32[lotusSharedIndex32(6)] = lotusWorkingSpace32[lotusSharedIndex32(14)]; \
        lotusWorkingSpace32[lotusSharedIndex32(7)] = lotusWorkingSpace32[lotusSharedIndex32(15)]; \
        lotusWorkingSpace32[lotusSharedIndex32(8)] = lotusWorkingSpace32[lotusSharedIndex32(0)] ^ lotusWorkingSpace32[lotusSharedIndex32(12)]; \
        lotusWorkingSpace32[lotusSharedIndex32(9)] = lotusWorkingSpace32[lotusSharedIndex32(1)] ^ lotusWorkingSpace32[lotusSharedIndex32(13)]; \
        lotusWorkingSpace32[lotusSharedIndex32(10)] = lotusWorkingSpace32[lotusSharedIndex32(2)] ^ lotusWorkingSpace32[lotusSharedIndex32(14)]; \
        lotusWorkingSpace32[lotusSharedIndex32(11)] = lotusWorkingSpace32[lotusSharedIndex32(3)] ^ lotusWorkingSpace32[lotusSharedIndex32(15)]; \
        X = 0; \
        for(i = 17; i > 0; i --) { \
        offset = 48; \
        for(j = 0; j < 48; j ++, offset --) X = (lotusWorkingSpace[lotusSharedIndex(j)] ^= S_S_BOX[(uint8_t)(offset + X)]); \
        } \
        offset = 48; \
        for(i = 0; i < 16; i ++, offset --)   X = (lotusWorkingSpace[lotusSharedIndex(i)] ^= S_S_BOX[(uint8_t)(offset + X)]); \
        a = lotusWorkingSpace32[lotusSharedIndex32(0)]; \
        b = lotusWorkingSpace32[lotusSharedIndex32(1)]; \
        c = lotusWorkingSpace32[lotusSharedIndex32(2)]; \
        d = lotusWorkingSpace32[lotusSharedIndex32(3)]; \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            if (!(deviceGlobalBitmap256kPlainLOTUS) || ((deviceGlobalBitmap256kPlainLOTUS[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                if (!(deviceGlobalBitmapAPlainLOTUS) || ((deviceGlobalBitmapAPlainLOTUS[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!deviceGlobalBitmapDPlainLOTUS || ((deviceGlobalBitmapDPlainLOTUS[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapCPlainLOTUS || ((deviceGlobalBitmapCPlainLOTUS[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapBPlainLOTUS || ((deviceGlobalBitmapBPlainLOTUS[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
            checkHashList128LE(a, b, c, d, b0, b1, b2, b3, \
                                    b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, \
                                    deviceGlobalFoundPasswordsPlainLOTUS, deviceGlobalFoundPasswordFlagsPlainLOTUS, \
                                    deviceGlobalHashlistAddressPlainLOTUS, numberOfHashesPlainLOTUS, \
                                    passwordLengthPlainLOTUS, MFN_PASSWORD_LOTUS); \
        }   }   }   }   }   } \
        if (charsetLengthsPlainLOTUS[1] == 0) { \
            makeMFNSingleIncrementors5 (sharedCharsetPlainLOTUS, sharedReverseCharsetPlainLOTUS, sharedCharsetLengthsPlainLOTUS); \
        } else { \
            makeMFNMultipleIncrementors5 (sharedCharsetPlainLOTUS, sharedReverseCharsetPlainLOTUS, sharedCharsetLengthsPlainLOTUS); \
        } \
        password_count++; \
    } \
    storePasswords32(deviceGlobalStartPasswords32PlainLOTUS, deviceNumberThreadsPlainLOTUS, pass_len); \
}

MAKE_MFN_LOTUS_KERNEL1_16LENGTH(1);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(2);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(3);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(4);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(5);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(6);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(7);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(8);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(9);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(10);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(11);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(12);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(13);
MAKE_MFN_LOTUS_KERNEL1_16LENGTH(14);
//MAKE_MFN_LOTUS_KERNEL1_16LENGTH(15);
//MAKE_MFN_LOTUS_KERNEL1_16LENGTH(16);

extern "C" cudaError_t MFNHashTypePlainCUDA_LOTUS_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_LOTUS_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_LOTUS_LaunchKernel()\n");

    //cudaPrintfInit();
    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_1 <<< Blocks, Threads >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_2 <<< Blocks, Threads >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_3 <<< Blocks, Threads >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_4 <<< Blocks, Threads >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_5 <<< Blocks, Threads >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_6 <<< Blocks, Threads >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_7 <<< Blocks, Threads >>> ();
            break;
        case 8:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_8 <<< Blocks, Threads >>> ();
            break;
        case 9:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_9 <<< Blocks, Threads >>> ();
            break;
        case 10:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_10 <<< Blocks, Threads >>> ();
            break;
        case 11:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_11 <<< Blocks, Threads >>> ();
            break;
        case 12:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_12 <<< Blocks, Threads >>> ();
            break;
        case 13:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_13 <<< Blocks, Threads >>> ();
            break;
        case 14:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_14 <<< Blocks, Threads >>> ();
            break;
        /*
         * TOOD: Figure out how to make this not build sm_10 kernels - needs >16k shared.
        case 15:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_15 <<< Blocks, Threads >>> ();
            break;
        case 16:
            MFNHashTypePlainCUDA_LOTUS_GeneratedKernel_16 <<< Blocks, Threads >>> ();
            break;
        */
        default:
            printf("Password length %d unsupported in CUDA!\n", passwordLength);
            exit(1);
            break;
            
    }
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();

    return cudaGetLastError();
}
