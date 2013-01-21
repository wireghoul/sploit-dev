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
 * This file implements DupMD5 multihash cracking.
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
#define MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_PASSLEN 48

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainDupMD5[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainDupMD5[MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainDupMD5[MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainDupMD5[MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainDupMD5;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainDupMD5;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainDupMD5;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainDupMD5;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainDupMD5;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainDupMD5;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainDupMD5;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainDupMD5;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainDupMD5;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainDupMD5;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainDupMD5;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainDupMD5;
__device__ __constant__ uint64_t deviceNumberThreadsPlainDupMD5;

__device__ inline void checkHashList128LEDuplicated(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, 
        uint8_t *deviceGlobalFoundPasswords, uint8_t *deviceGlobalFoundPasswordFlags,
        uint8_t *deviceGlobalHashlistAddress, uint64_t numberOfHashes,
        uint8_t passwordLength, uint8_t algorithmType) {
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
                    switch (passwordLength) {
                        case 16:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 15] = (b3 >> 24) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 15 + passwordLength] = (b3 >> 24) & 0xff;
                        case 15:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 14] = (b3 >> 16) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 14 + passwordLength] = (b3 >> 16) & 0xff;
                        case 14:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 13] = (b3 >> 8) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 13 + passwordLength] = (b3 >> 8) & 0xff;
                        case 13:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 12] = (b3 >> 0) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 12 + passwordLength] = (b3 >> 0) & 0xff;
                        case 12:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 11] = (b2 >> 24) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 11 + passwordLength] = (b2 >> 24) & 0xff;
                        case 11:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 10] = (b2 >> 16) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 10 + passwordLength] = (b2 >> 16) & 0xff;
                        case 10:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 9] = (b2 >> 8) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 9 + passwordLength] = (b2 >> 8) & 0xff;
                        case 9:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 8] = (b2 >> 0) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 8 + passwordLength] = (b2 >> 0) & 0xff;
                        case 8:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 7] = (b1 >> 24) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 7 + passwordLength] = (b1 >> 24) & 0xff;
                        case 7:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 6] = (b1 >> 16) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 6 + passwordLength] = (b1 >> 16) & 0xff;
                        case 6:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 5] = (b1 >> 8) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 5 + passwordLength] = (b1 >> 8) & 0xff;
                        case 5:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 4] = (b1 >> 0) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 4 + passwordLength] = (b1 >> 0) & 0xff;
                        case 4:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 3] = (b0 >> 24) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 3 + passwordLength] = (b0 >> 24) & 0xff;
                        case 3:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 2] = (b0 >> 16) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 2 + passwordLength] = (b0 >> 16) & 0xff;
                        case 2:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 1] = (b0 >> 8) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 1 + passwordLength] = (b0 >> 8) & 0xff;
                        case 1:
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 0] = (b0 >> 0) & 0xff;
                            deviceGlobalFoundPasswords[search_index * (passwordLength * 2) + 0 + passwordLength] = (b0 >> 0) & 0xff;
                    }
                    deviceGlobalFoundPasswordFlags[search_index] = (uint8_t) algorithmType;
                }
            }
        }
        search_index++;
    }
}



#define DuplicatePassword(pass_length) { \
if (pass_length == 1) { \
    b0 = (b0 & 0x000000ff) | ((b0 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 2) { \
    b0 = (b0 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b1 = 0x00000080; \
} else if (pass_length == 3) {\
    b0 = (b0 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b1 = ((b0 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 4) {\
    b1 = b0; \
    b2 = 0x00000080; \
} else if (pass_length == 5) {\
    b1 = (b1 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b2 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 6) {\
    b1 = (b1 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b2 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b3 = 0x00000080; \
} else if (pass_length == 7) {\
    b1 = (b1 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b2 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b3 = ((b1 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 8) {\
    b2 = b0; \
    b3 = b1; \
    b4 = 0x00000080; \
} else if (pass_length == 9) {\
    b2 = (b2 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b3 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b4 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 10) {\
    b2 = (b2 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b3 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b4 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b5 = 0x00000080; \
} else if (pass_length == 11) {\
    b2 = (b2 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b3 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b4 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b5 = ((b2 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 12) {\
    b3 = b0; \
    b4 = b1; \
    b5 = b2; \
    b6 = 0x00000080; \
} else if (pass_length == 13) {\
    b3 = (b3 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b4 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b5 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b6 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 14) {\
    b3 = (b3 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b4 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b5 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b6 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b7 = 0x00000080; \
} else if (pass_length == 15) {\
    b3 = (b3 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b4 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b5 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b6 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b7 = ((b3 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 16) {\
    b4 = b0; \
    b5 = b1; \
    b6 = b2; \
    b7 = b3; \
    b8 = 0x00000080; \
} else if (pass_length == 17) {\
    b4 = (b4 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b5 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b6 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b7 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x00ffffff) << 8); \
    b8 = ((b3 & 0xff000000) >> 24) | ((b4 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 18) {\
    b4 = (b4 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b5 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b6 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b7 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b8 = ((b3 & 0xffff0000) >> 16) | ((b4 & 0x0000ffff) << 16); \
    b9 = 0x00000080; \
} else if (pass_length == 19) {\
    b4 = (b4 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b5 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b6 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b7 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b8 = ((b3 & 0xffffff00) >> 8) | ((b4 & 0x000000ff) << 24); \
    b9 = ((b4 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 20) {\
    b5 = b0; \
    b6 = b1; \
    b7 = b2; \
    b8 = b3; \
    b9 = b4; \
    b10 = 0x00000080; \
} else if (pass_length == 21) {\
    b5 = (b5 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b6 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b7 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b8 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x00ffffff) << 8); \
    b9 = ((b3 & 0xff000000) >> 24) | ((b4 & 0x00ffffff) << 8); \
    b10 = ((b4 & 0xff000000) >> 24) | ((b5 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 22) {\
    b5 = (b5 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b6 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b7 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b8 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b9 = ((b3 & 0xffff0000) >> 16) | ((b4 & 0x0000ffff) << 16); \
    b10 = ((b4 & 0xffff0000) >> 16) | ((b5 & 0x0000ffff) << 16); \
    b11 = 0x00000080; \
} else if (pass_length == 23) {\
    b5 = (b5 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b6 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b7 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b8 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b9 = ((b3 & 0xffffff00) >> 8) | ((b4 & 0x000000ff) << 24); \
    b10 = ((b4 & 0xffffff00) >> 8) | ((b5 & 0x000000ff) << 24); \
    b11 = ((b5 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 24) {\
    b6 = b0; \
    b7 = b1; \
    b8 = b2; \
    b9 = b3; \
    b10 = b4; \
    b11 = b5; \
    b12 = 0x00000080; \
} else if (pass_length == 25) {\
    b6 = (b6 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b7 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b8 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b9 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x00ffffff) << 8); \
    b10 = ((b3 & 0xff000000) >> 24) | ((b4 & 0x00ffffff) << 8); \
    b11 = ((b4 & 0xff000000) >> 24) | ((b5 & 0x00ffffff) << 8); \
    b12 = ((b5 & 0xff000000) >> 24) | ((b6 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 26) {\
    b6 = (b6 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b7 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b8 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b9 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b10 = ((b3 & 0xffff0000) >> 16) | ((b4 & 0x0000ffff) << 16); \
    b11 = ((b4 & 0xffff0000) >> 16) | ((b5 & 0x0000ffff) << 16); \
    b12 = ((b5 & 0xffff0000) >> 16) | ((b6 & 0x0000ffff) << 16); \
    b13 = 0x00000080; \
} else if (pass_length == 27) {\
    b6 = (b6 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b7 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b8 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b9 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b10 = ((b3 & 0xffffff00) >> 8) | ((b4 & 0x000000ff) << 24); \
    b11 = ((b4 & 0xffffff00) >> 8) | ((b5 & 0x000000ff) << 24); \
    b12 = ((b5 & 0xffffff00) >> 8) | ((b6 & 0x000000ff) << 24); \
    b13 = ((b6 & 0x00ffff00) >> 8) | 0x00800000; \
}\
}


#define MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainDupMD5[MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainDupMD5[MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t               sharedCharsetLengthsPlainDupMD5[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainDupMD5; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainDupMD5; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainDupMD5; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainDupMD5; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainDupMD5; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_DUP_MD5_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainDupMD5[a] = charsetLengthsPlainDupMD5[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
    } \
    syncthreads(); \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    b14 = pass_len * 8 * 2; \
    loadPasswords32(deviceGlobalStartPasswords32PlainDupMD5, deviceNumberThreadsPlainDupMD5, pass_len); \
    while (password_count < deviceNumberStepsToRunPlainDupMD5) { \
        /*cuPrintf("b0 init: %08x %08x %08x\n", b0, b1, b2);*/ \
        DuplicatePassword(pass_len); \
        /*cuPrintf("b0: %08x %08x %08x b14: %08x\n", b0, b1, b2, b14);*/ \
        MD5_FIRST_4_ROUNDS(); \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            if (!(deviceGlobalBitmapAPlainDupMD5) || ((deviceGlobalBitmapAPlainDupMD5[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                if (!deviceGlobalBitmapDPlainDupMD5 || ((deviceGlobalBitmapDPlainDupMD5[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                    if (!deviceGlobalBitmapCPlainDupMD5 || ((deviceGlobalBitmapCPlainDupMD5[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapBPlainDupMD5 || ((deviceGlobalBitmapBPlainDupMD5[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                            checkHashList128LEDuplicated(a, b, c, d, b0, b1, b2, b3, \
                                deviceGlobalFoundPasswordsPlainDupMD5, deviceGlobalFoundPasswordFlagsPlainDupMD5, \
                                deviceGlobalHashlistAddressPlainDupMD5, numberOfHashesPlainDupMD5, \
                                passwordLengthPlainDupMD5, MFN_PASSWORD_SINGLE_MD5); \
        }   }   }   }   }\
        if (charsetLengthsPlainDupMD5[1] == 0) { \
                makeMFNSingleIncrementors##pass_len (sharedCharsetPlainDupMD5, sharedReverseCharsetPlainDupMD5, sharedCharsetLengthsPlainDupMD5); \
        } else { \
                makeMFNMultipleIncrementors##pass_len (sharedCharsetPlainDupMD5, sharedReverseCharsetPlainDupMD5, sharedCharsetLengthsPlainDupMD5); \
        } \
        password_count++; \
    } \
    storePasswords32(deviceGlobalStartPasswords32PlainDupMD5, deviceNumberThreadsPlainDupMD5, pass_len); \
}

MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(1);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(2);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(3);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(4);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(5);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(6);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(7);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(8);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(9);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(10);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(11);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(12);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(13);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(14);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(15);
MAKE_MFN_DUP_MD5_KERNEL1_8LENGTH(16);

extern "C" cudaError_t MFNHashTypePlainCUDA_DupMD5_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_DupMD5_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_MD5_LaunchKernel()\n");

    //cudaPrintfInit();
    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_1 <<< Blocks, Threads >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_2 <<< Blocks, Threads >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_3 <<< Blocks, Threads >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_4 <<< Blocks, Threads >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_5 <<< Blocks, Threads >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_6 <<< Blocks, Threads >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_7 <<< Blocks, Threads >>> ();
            break;
        case 8:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_8 <<< Blocks, Threads >>> ();
            break;
        case 9:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_9 <<< Blocks, Threads >>> ();
            break;
        case 10:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_10 <<< Blocks, Threads >>> ();
            break;
        case 11:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_11 <<< Blocks, Threads >>> ();
            break;
        case 12:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_12 <<< Blocks, Threads >>> ();
            break;
        case 13:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_13 <<< Blocks, Threads >>> ();
            break;
        case 14:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_14 <<< Blocks, Threads >>> ();
            break;
        case 15:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_15 <<< Blocks, Threads >>> ();
            break;
        case 16:
            MFNHashTypePlainCUDA_DupMD5_GeneratedKernel_16 <<< Blocks, Threads >>> ();
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
