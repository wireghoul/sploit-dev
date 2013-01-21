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

#include "CUDA_Common/cuPrintf.cu"

#include "MFN_CUDA_device/MFN_CUDA_incrementors.h"
#include "MFN_CUDA_device/MFN_CUDA_Common.h"
#include "MFN_CUDA_device/MFN_CUDA_MD5.h"
#include "MFN_CUDA_device/MFN_CUDA_ConstantCopyDefines.h"

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
#define MFN_HASH_TYPE_PLAIN_CUDA_IPBWL_MAX_PASSLEN 128

// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainIPBWL[8192];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainIPBWL;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainIPBWL;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainIPBWL;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainIPBWL;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainIPBWL;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainIPBWL;
__device__ __constant__ uint8_t *deviceGlobalBitmap256kPlainIPBWL;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainIPBWL;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainIPBWL;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainIPBWL;
__device__ __constant__ uint64_t deviceNumberThreadsPlainIPBWL;
__device__ __constant__ uint32_t deviceStartStepPlainIPBWL;

// Salted hash data
__device__ __constant__ uint32_t *deviceGlobalSaltLengthsIPBWL;
__device__ __constant__ uint32_t *deviceGlobalSaltValuesIPBWL;
__device__ __constant__ uint32_t deviceNumberOfSaltValuesIPBWL;
__device__ __constant__ uint32_t deviceStartingSaltOffsetIPBWL;

// Wordlist data
__device__ __constant__ uint32_t *deviceGlobalWordlistDataIPBWL;
__device__ __constant__ uint8_t  *deviceGlobalWordlistLengthsIPBWL;
__device__ __constant__ uint32_t deviceNumberWordsIPBWL;
__device__ __constant__ uint8_t  deviceWordlistBlocksIPBWL;

__constant__ char hexLookupValuesIPBWL[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

#define AddHashCharacterAsString_LE_LE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 8) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  hashValue = hashValue >> 16; \
}

#define IPB_EXPAND_PASSWORD_TO_ASCII() { \
    b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    AddHashCharacterAsString_LE_LE(hashLookup, a, b8, temp); \
    AddHashCharacterAsString_LE_LE(hashLookup, a, b9, temp); \
    AddHashCharacterAsString_LE_LE(hashLookup, b, b10, temp); \
    AddHashCharacterAsString_LE_LE(hashLookup, b, b11, temp); \
    AddHashCharacterAsString_LE_LE(hashLookup, c, b12, temp); \
    AddHashCharacterAsString_LE_LE(hashLookup, c, b13, temp); \
    AddHashCharacterAsString_LE_LE(hashLookup, d, b14, temp); \
    AddHashCharacterAsString_LE_LE(hashLookup, d, b15, temp); \
}

#define CopyFoundPasswordToMemoryFromWordlistLE(dfp, dfpf, passwordStep) { \
    uint8_t passwordLength = deviceGlobalWordlistLengthsIPBWL[(thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL))]; \
    uint32_t wordlistBlock; \
    for (uint32_t i = 0; i < passwordLength; i++) { \
        if ((i % 4) == 0) { \
            wordlistBlock = deviceGlobalWordlistDataIPBWL[thread_index + (i / 4) * deviceNumberWordsIPBWL + \
                passwordStep * deviceNumberThreadsPlainIPBWL]; \
        } \
        dfp[search_index * MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN + i] = (wordlistBlock >> ((i % 4) * 8)) & 0xff; \
    } \
    dfpf[search_index] = (unsigned char) 1; \
}

__device__ inline void checkHashList128LEWordlist(
        const uint32_t &a, const uint32_t &b, const uint32_t &c, const uint32_t &d,
        uint8_t *deviceGlobalFoundPasswords, uint8_t *deviceGlobalFoundPasswordFlags,
        const uint8_t *deviceGlobalHashlistAddress, const uint64_t &numberOfHashes,
        uint8_t algorithmType, const uint32_t &passwordStep) {
    uint32_t search_high, search_low, search_index, current_hash_value;
    uint32_t *DEVICE_Hashes_32 = (uint32_t *) deviceGlobalHashlistAddress;
    search_high = numberOfHashes;
    search_low = 0;
    while (search_low < search_high) {
        // Midpoint between search_high and search_low
        search_index = search_low + (search_high - search_low) / 2;
        current_hash_value = DEVICE_Hashes_32[4 * search_index];
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

    while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
        search_index--;
    }
    while ((a == DEVICE_Hashes_32[search_index * 4])) {
        if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
            if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
                if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {
                    // Copy the password to the correct location.
                    CopyFoundPasswordToMemoryFromWordlistLE(deviceGlobalFoundPasswords, deviceGlobalFoundPasswordFlags, passwordStep);
                }
            }
        }
        search_index++;
    }
}

#define CUDA_COPY_8KB_BITMAP(constantSource, sharedDestination) { \
    uint64_t *constantBitmap64 = (uint64_t *)constantSource; \
    uint64_t *sharedBitmap64 = (uint64_t *)sharedDestination; \
    for (a = 0; a < 8192 / 8; a++) { \
        sharedBitmap64[a] = constantBitmap64[a]; \
    } \
}

#define CUDA_COPY_HEX2BIN_VALUES(constantSource, sharedDestination) { \
    for (a = 0; a < 256; a++) { \
        sharedDestination[a][0] = constantSource[a / 16]; \
        sharedDestination[a][1] = constantSource[a % 16]; \
    } \
}

/**
 * Load a data block with the specified offset.  This will load the b0-b13 range
 * that is common to all block loads.
 * @param wordlistAddress The address of the wordlist array
 * @param blocksToLoad How many blocks to load from the wordlist
 * @param dataOffset Block offset - 0 for the first block, 16 for the 2nd, etc.
 * @param numberWords The total number of words in the wordlist
 * @param passwordStep The word step the thread is on.
 * @param numberThreads The number of threads in the execution environment.
 */
#define CUDA_LOAD_LE_MAIN_DATA_BLOCK(wordlistAddress, blocksToLoad, dataOffset,\
        numberWords, passwordStep, numberThreads) { \
        b0 = wordlistAddress[((0 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        if (blocksToLoad >= 2) { \
            b1 = wordlistAddress[((1 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 3) { \
            b2 = wordlistAddress[((2 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 4) { \
            b3 = wordlistAddress[((3 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 5) { \
            b4 = wordlistAddress[((4 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 6) { \
            b5 = wordlistAddress[((5 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 7) { \
            b6 = wordlistAddress[((6 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 8) { \
            b7 = wordlistAddress[((7 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 9) { \
            b8 = wordlistAddress[((8 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 10) { \
            b9 = wordlistAddress[((9 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 11) { \
            b10 = wordlistAddress[((10 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 12) { \
            b11 = wordlistAddress[((11 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 13) { \
            b12 = wordlistAddress[((12 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
        if (blocksToLoad >= 14) { \
            b13 = wordlistAddress[((13 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
}


/**
 * Load a data block with the specified offset.  This will load the b14-b15 that
 * is needed for the longer plain lengths.
 * 
 * @param wordlistAddress The address of the wordlist array
 * @param blocksToLoad How many blocks to load from the wordlist
 * @param dataOffset Block offset - 0 for the first block, 16 for the 2nd, etc.
 * @param numberWords The total number of words in the wordlist
 * @param passwordStep The word step the thread is on.
 * @param numberThreads The number of threads in the execution environment.
 */
#define CUDA_LOAD_LE_FINAL_DATA_BLOCK(wordlistAddress, blocksToLoad, dataOffset,\
        numberWords, passwordStep, numberThreads) { \
        b14 = wordlistAddress[((14 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        if (blocksToLoad >= 15) { \
            b15 = wordlistAddress[((15 + dataOffset) * numberWords + \
                passwordStep * numberThreads) + thread_index]; \
        } \
}

/**
 * Copy 32 bytes of salt (8 words) from the correct global offset into the
 * shared memory location 
 */
#define CUDA_LOAD_32_BYTE_SALT_TO_SHARED(globalSalts, sharedSalts, \
        numberSalts, saltIndex) { \
    sharedSalts[0] = globalSalts[(0 * numberSalts) + saltIndex]; \
    sharedSalts[1] = globalSalts[(1 * numberSalts) + saltIndex]; \
    sharedSalts[2] = globalSalts[(2 * numberSalts) + saltIndex]; \
    sharedSalts[3] = globalSalts[(3 * numberSalts) + saltIndex]; \
    sharedSalts[4] = globalSalts[(4 * numberSalts) + saltIndex]; \
    sharedSalts[5] = globalSalts[(5 * numberSalts) + saltIndex]; \
    sharedSalts[6] = globalSalts[(6 * numberSalts) + saltIndex]; \
    sharedSalts[7] = globalSalts[(7 * numberSalts) + saltIndex]; \
}


__global__ void MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B1_14 () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t prev_a, prev_b, prev_c, prev_d;
    uint32_t password_count = 0, passwordStep, saltIndex, temp; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint8_t hashLookup[256][2];
    __shared__ uint32_t saltPrefetch[8];
    if (threadIdx.x == 0) { \
        CUDA_COPY_8KB_BITMAP(constantBitmapAPlainIPBWL, sharedBitmap); \
        CUDA_COPY_HEX2BIN_VALUES(hexLookupValuesIPBWL, hashLookup); \
    } \

    passwordStep = deviceStartStepPlainIPBWL;
    syncthreads(); \
    saltIndex = deviceStartingSaltOffsetIPBWL;
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
        CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep *
                deviceNumberThreadsPlainIPBWL + thread_index];
        MD5_FULL_HASH();
        //cuPrintf("word hash: %08x %08x %08x %08x\n", a, b, c, d);
        IPB_EXPAND_PASSWORD_TO_ASCII();
        //cuPrintf("b8-b15: %08x %08x %08x %08x %08x %08x %08x %08x\n", 
        //        b8, b9, b10, b11, b12, b13, b14, b15);
    }
 
    while (password_count < deviceNumberStepsToRunPlainIPBWL) { \
        //cuPrintf("saltIndex: %u\n", saltIndex);
        if (threadIdx.x == 0) {
            CUDA_LOAD_32_BYTE_SALT_TO_SHARED(deviceGlobalSaltValuesIPBWL, \
                    saltPrefetch, deviceNumberOfSaltValuesIPBWL, saltIndex);
        }
        syncthreads();
        b0 = saltPrefetch[0]; b1 = saltPrefetch[1]; b2 = saltPrefetch[2]; b3 = saltPrefetch[3];
        b4 = saltPrefetch[4]; b5 = saltPrefetch[5]; b6 = saltPrefetch[6]; b7 = saltPrefetch[7];
        if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            //cuPrintf("Hash results: %08x %08x %08x %08x\n", a, b, c, d);
            if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
                if (!(deviceGlobalBitmap256kPlainIPBWL) || ((deviceGlobalBitmap256kPlainIPBWL[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!(deviceGlobalBitmapAPlainIPBWL) || ((deviceGlobalBitmapAPlainIPBWL[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapDPlainIPBWL || ((deviceGlobalBitmapDPlainIPBWL[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapCPlainIPBWL || ((deviceGlobalBitmapCPlainIPBWL[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                                if (!deviceGlobalBitmapBPlainIPBWL || ((deviceGlobalBitmapBPlainIPBWL[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LEWordlist(a, b, c, d,
                                    deviceGlobalFoundPasswordsPlainIPBWL, deviceGlobalFoundPasswordFlagsPlainIPBWL,
                                    deviceGlobalHashlistAddressPlainIPBWL, numberOfHashesPlainIPBWL,
                                    1, passwordStep);
            }   }   }   }   }   } \
        }

        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValuesIPBWL) {
            //cuPrintf("Resetting salt index to 0.\n");
            saltIndex = 0;
            passwordStep++;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
            if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
                CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep * deviceNumberThreadsPlainIPBWL + thread_index];
                //cuPrintf("Loaded: %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4);
                MD5_FULL_HASH();
                //cuPrintf("word hash: %08x %08x %08x %08x\n", a, b, c, d);
                IPB_EXPAND_PASSWORD_TO_ASCII();
                //cuPrintf("b8-b15: %08x %08x %08x %08x %08x %08x %08x %08x\n", 
                //        b8, b9, b10, b11, b12, b13, b14, b15);
            }
        }
        password_count++;
    } \
}

__global__ void MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B15_16 () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t prev_a, prev_b, prev_c, prev_d;
    uint32_t password_count = 0, passwordStep, saltIndex, temp; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint8_t hashLookup[256][2];
    __shared__ uint32_t saltPrefetch[8];
    if (threadIdx.x == 0) { \
        CUDA_COPY_8KB_BITMAP(constantBitmapAPlainIPBWL, sharedBitmap); \
        CUDA_COPY_HEX2BIN_VALUES(hexLookupValuesIPBWL, hashLookup); \
    } \
    passwordStep = deviceStartStepPlainIPBWL;
    syncthreads(); \
    saltIndex = deviceStartingSaltOffsetIPBWL;
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
        CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        MD5_FULL_HASH();
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep *
                deviceNumberThreadsPlainIPBWL + thread_index];
        MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPBWL) { \
        if (threadIdx.x == 0) {
            CUDA_LOAD_32_BYTE_SALT_TO_SHARED(deviceGlobalSaltValuesIPBWL, \
                    saltPrefetch, deviceNumberOfSaltValuesIPBWL, saltIndex);
        }
        syncthreads();
        b0 = saltPrefetch[0]; b1 = saltPrefetch[1]; b2 = saltPrefetch[2]; b3 = saltPrefetch[3];
        b4 = saltPrefetch[4]; b5 = saltPrefetch[5]; b6 = saltPrefetch[6]; b7 = saltPrefetch[7];
        if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
                if (!(deviceGlobalBitmap256kPlainIPBWL) || ((deviceGlobalBitmap256kPlainIPBWL[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!(deviceGlobalBitmapAPlainIPBWL) || ((deviceGlobalBitmapAPlainIPBWL[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapDPlainIPBWL || ((deviceGlobalBitmapDPlainIPBWL[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapCPlainIPBWL || ((deviceGlobalBitmapCPlainIPBWL[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                                if (!deviceGlobalBitmapBPlainIPBWL || ((deviceGlobalBitmapBPlainIPBWL[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LEWordlist(a, b, c, d,
                                    deviceGlobalFoundPasswordsPlainIPBWL, deviceGlobalFoundPasswordFlagsPlainIPBWL,
                                    deviceGlobalHashlistAddressPlainIPBWL, numberOfHashesPlainIPBWL,
                                    1, passwordStep);
            }   }   }   }   }   } \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValuesIPBWL) {
            saltIndex = 0;
            passwordStep++;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
            if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
                CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep *
                        deviceNumberThreadsPlainIPBWL + thread_index];
                MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    } \
}


__global__ void MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B17_30 () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t prev_a, prev_b, prev_c, prev_d;
    uint32_t password_count = 0, passwordStep, saltIndex, temp; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint8_t hashLookup[256][2];
    __shared__ uint32_t saltPrefetch[8];
    if (threadIdx.x == 0) { \
        CUDA_COPY_8KB_BITMAP(constantBitmapAPlainIPBWL, sharedBitmap); \
        CUDA_COPY_HEX2BIN_VALUES(hexLookupValuesIPBWL, hashLookup); \
    } \
    passwordStep = deviceStartStepPlainIPBWL;
    syncthreads(); \
    saltIndex = deviceStartingSaltOffsetIPBWL;
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
        CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        MD5_FULL_HASH();
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
        CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                (deviceWordlistBlocksIPBWL - 16), 16, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep *
                deviceNumberThreadsPlainIPBWL + thread_index];
        MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPBWL) { \
        if (threadIdx.x == 0) {
            CUDA_LOAD_32_BYTE_SALT_TO_SHARED(deviceGlobalSaltValuesIPBWL, \
                    saltPrefetch, deviceNumberOfSaltValuesIPBWL, saltIndex);
        }
        syncthreads();
        b0 = saltPrefetch[0]; b1 = saltPrefetch[1]; b2 = saltPrefetch[2]; b3 = saltPrefetch[3];
        b4 = saltPrefetch[4]; b5 = saltPrefetch[5]; b6 = saltPrefetch[6]; b7 = saltPrefetch[7];
        if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
                if (!(deviceGlobalBitmap256kPlainIPBWL) || ((deviceGlobalBitmap256kPlainIPBWL[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!(deviceGlobalBitmapAPlainIPBWL) || ((deviceGlobalBitmapAPlainIPBWL[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapDPlainIPBWL || ((deviceGlobalBitmapDPlainIPBWL[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapCPlainIPBWL || ((deviceGlobalBitmapCPlainIPBWL[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                                if (!deviceGlobalBitmapBPlainIPBWL || ((deviceGlobalBitmapBPlainIPBWL[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LEWordlist(a, b, c, d,
                                    deviceGlobalFoundPasswordsPlainIPBWL, deviceGlobalFoundPasswordFlagsPlainIPBWL,
                                    deviceGlobalHashlistAddressPlainIPBWL, numberOfHashesPlainIPBWL,
                                    1, passwordStep);
            }   }   }   }   }   } \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValuesIPBWL) {
            saltIndex = 0;
            passwordStep++;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
            if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
                CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
                CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        (deviceWordlistBlocksIPBWL - 16), 16, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep *
                        deviceNumberThreadsPlainIPBWL + thread_index];
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    } \
}

__global__ void MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B31_32 () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t prev_a, prev_b, prev_c, prev_d;
    uint32_t password_count = 0, passwordStep, saltIndex, temp; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    __shared__ uint8_t hashLookup[256][2];
    __shared__ uint32_t saltPrefetch[8];
    if (threadIdx.x == 0) { \
        CUDA_COPY_8KB_BITMAP(constantBitmapAPlainIPBWL, sharedBitmap); \
        CUDA_COPY_HEX2BIN_VALUES(hexLookupValuesIPBWL, hashLookup); \
    } \
    passwordStep = deviceStartStepPlainIPBWL;
    syncthreads(); \
    saltIndex = deviceStartingSaltOffsetIPBWL;
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
        CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        MD5_FULL_HASH();
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
        CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                (deviceWordlistBlocksIPBWL - 16), 16, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                (deviceWordlistBlocksIPBWL - 16), 16, deviceNumberWordsIPBWL, \
                passwordStep, deviceNumberThreadsPlainIPBWL);
        MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep *
                deviceNumberThreadsPlainIPBWL + thread_index];
        MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPBWL) { \
        if (threadIdx.x == 0) {
            CUDA_LOAD_32_BYTE_SALT_TO_SHARED(deviceGlobalSaltValuesIPBWL, \
                    saltPrefetch, deviceNumberOfSaltValuesIPBWL, saltIndex);
        }
        syncthreads();
        b0 = saltPrefetch[0]; b1 = saltPrefetch[1]; b2 = saltPrefetch[2]; b3 = saltPrefetch[3];
        b4 = saltPrefetch[4]; b5 = saltPrefetch[5]; b6 = saltPrefetch[6]; b7 = saltPrefetch[7];
        if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
                if (!(deviceGlobalBitmap256kPlainIPBWL) || ((deviceGlobalBitmap256kPlainIPBWL[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!(deviceGlobalBitmapAPlainIPBWL) || ((deviceGlobalBitmapAPlainIPBWL[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapDPlainIPBWL || ((deviceGlobalBitmapDPlainIPBWL[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapCPlainIPBWL || ((deviceGlobalBitmapCPlainIPBWL[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                                if (!deviceGlobalBitmapBPlainIPBWL || ((deviceGlobalBitmapBPlainIPBWL[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                checkHashList128LEWordlist(a, b, c, d,
                                    deviceGlobalFoundPasswordsPlainIPBWL, deviceGlobalFoundPasswordFlagsPlainIPBWL,
                                    deviceGlobalHashlistAddressPlainIPBWL, numberOfHashesPlainIPBWL,
                                    1, passwordStep);
            }   }   }   }   }   } \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValuesIPBWL) {
            saltIndex = 0;
            passwordStep++;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
            if ((thread_index + (passwordStep * deviceNumberThreadsPlainIPBWL)) < deviceNumberWordsIPBWL) {
                CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        deviceWordlistBlocksIPBWL, 0, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0;
                CUDA_LOAD_LE_MAIN_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        (deviceWordlistBlocksIPBWL - 16), 16, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                CUDA_LOAD_LE_FINAL_DATA_BLOCK(deviceGlobalWordlistDataIPBWL, \
                        (deviceWordlistBlocksIPBWL - 16), 16, deviceNumberWordsIPBWL, \
                        passwordStep, deviceNumberThreadsPlainIPBWL);
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b14 = 8 * deviceGlobalWordlistLengthsIPBWL[passwordStep *
                        deviceNumberThreadsPlainIPBWL + thread_index];
                MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    } \
}

/**
 * Copy data by symbol ID to the device constant space based on the requested
 * transfer.
 */
extern "C" cudaError_t MFNHashTypeSaltedCUDA_IPBWL_CopyValueToConstantById(
        uint32_t symbolId, void *hostDataAddress, size_t bytesToCopy) {
    switch (symbolId) {
        // 8kb bitmap for shared memory
        case MFN_CUDA_CONSTANT_BITMAP_A:
            return cudaMemcpyToSymbol(constantBitmapAPlainIPBWL,
                    hostDataAddress, bytesToCopy);

        case MFN_CUDA_DEVICE_STEPS_TO_RUN:
            return cudaMemcpyToSymbol(deviceNumberStepsToRunPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_NUMBER_THREADS:
            return cudaMemcpyToSymbol(deviceNumberThreadsPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_START_STEP:
            return cudaMemcpyToSymbol(deviceStartStepPlainIPBWL,
                    hostDataAddress, bytesToCopy);
            
        // Global bitmaps and 256kb bitmaps
        case MFN_CUDA_GLOBAL_BITMAP_A:
            return cudaMemcpyToSymbol(deviceGlobalBitmapAPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_GLOBAL_BITMAP_B:
            return cudaMemcpyToSymbol(deviceGlobalBitmapBPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_GLOBAL_BITMAP_C:
            return cudaMemcpyToSymbol(deviceGlobalBitmapCPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_GLOBAL_BITMAP_D:
            return cudaMemcpyToSymbol(deviceGlobalBitmapDPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_GLOBAL_BITMAP_256KB_A:
            return cudaMemcpyToSymbol(deviceGlobalBitmap256kPlainIPBWL,
                    hostDataAddress, bytesToCopy);
            
        // Found password list and flags
        case MFN_CUDA_GLOBAL_FOUND_PASSWORDS:
            return cudaMemcpyToSymbol(deviceGlobalFoundPasswordsPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_GLOBAL_FOUND_PASSWORD_FLAGS:
            return cudaMemcpyToSymbol(deviceGlobalFoundPasswordFlagsPlainIPBWL,
                    hostDataAddress, bytesToCopy);
            
        // Global hashlist data
        case MFN_CUDA_GLOBAL_HASHLIST_ADDRESS:
            return cudaMemcpyToSymbol(deviceGlobalHashlistAddressPlainIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_NUMBER_OF_HASHES:
            return cudaMemcpyToSymbol(numberOfHashesPlainIPBWL,
                    hostDataAddress, bytesToCopy);
            
        // Salt data
        case MFN_CUDA_DEVICE_SALT_DATA:
            return cudaMemcpyToSymbol(deviceGlobalSaltValuesIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_SALT_LENGTHS:
            return cudaMemcpyToSymbol(deviceGlobalSaltLengthsIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_NUMBER_SALTS:
            return cudaMemcpyToSymbol(deviceNumberOfSaltValuesIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_STARTING_SALT_OFFSET:
            return cudaMemcpyToSymbol(deviceStartingSaltOffsetIPBWL,
                    hostDataAddress, bytesToCopy);
        
        // Wordlist data
        case MFN_CUDA_DEVICE_WORDLIST_DATA:
            return cudaMemcpyToSymbol(deviceGlobalWordlistDataIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_WORDLIST_LENGTHS:
            return cudaMemcpyToSymbol(deviceGlobalWordlistLengthsIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_NUMBER_WORDS:
            return cudaMemcpyToSymbol(deviceNumberWordsIPBWL,
                    hostDataAddress, bytesToCopy);
        case MFN_CUDA_DEVICE_BLOCKS_PER_WORD:
            return cudaMemcpyToSymbol(deviceWordlistBlocksIPBWL,
                    hostDataAddress, bytesToCopy);
            
        default:
            return cudaErrorInvalidValue;
    }
}



extern "C" cudaError_t MFNHashTypeSaltedCUDA_IPBWL_LaunchKernel(uint32_t wordlistLengthBlocks, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypeSaltedCUDA_IPBWL_LaunchKernel()\n");

    cudaPrintfInit();
    switch (wordlistLengthBlocks) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
            MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B1_14 <<< Blocks, Threads >>> ();
            break;
        case 15:
        case 16:
            MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B15_16 <<< Blocks, Threads >>> ();
            break;
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
            MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B17_30 <<< Blocks, Threads >>> ();
            break;
        case 31:
        case 32:
            MFNHashTypeSaltedCUDA_IPBWL_GeneratedKernel_B31_32 <<< Blocks, Threads >>> ();
            break;
        default:
            printf("Password length %d unsupported!\n", wordlistLengthBlocks);
            exit(1);
            break;
    }
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();

    return cudaGetLastError();
}
