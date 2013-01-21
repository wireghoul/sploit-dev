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
 * This file implements SHA256 multihash cracking.
 */

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

//#include "CUDA_Common/cuPrintf.cu"

#include "MFN_CUDA_device/MFN_CUDA_SHA_incrementors.h"
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

/**
 * The maximum password length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_PASSLEN 48

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH 128


// Define the constant types used by the kernels here.
__device__ __constant__ __align__(16) uint8_t  constantBitmapAPlainSHA256[8192];
__device__ __constant__ __align__(16) uint8_t deviceCharsetPlainSHA256[MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_PASSLEN];
__device__ __constant__ __align__(16) uint8_t deviceReverseCharsetPlainSHA256[MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * \
    MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_PASSLEN];
__device__ __constant__ uint8_t charsetLengthsPlainSHA256[MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_PASSLEN];

/**
 * Constant parameters go here instead of getting passed as kernel arguments.
 * This allows for faster accesses (as they are cached, and all threads will
 * be accessing the same element), and also reduces the shared memory usage,
 * which may allow for better occupancy in the future.  The kernels will load
 * these as needed, and theoretically will not need registers for some of them,
 * which will help reduce the register pressure on kernels.  Hopefully.
 */

// Password length.  Needed for some offset calculations.
__device__ __constant__ uint8_t passwordLengthPlainSHA256;

// Number of hashes present in memory.
__device__ __constant__ uint64_t numberOfHashesPlainSHA256;

// Address of the hashlist in global memory.
__device__ __constant__ uint8_t *deviceGlobalHashlistAddressPlainSHA256;

// Addresses of the various global bitmaps.
__device__ __constant__ uint8_t *deviceGlobalBitmapAPlainSHA256;
__device__ __constant__ uint8_t *deviceGlobalBitmapBPlainSHA256;
__device__ __constant__ uint8_t *deviceGlobalBitmapCPlainSHA256;
__device__ __constant__ uint8_t *deviceGlobalBitmapDPlainSHA256;
__device__ __constant__ uint8_t *deviceGlobalBitmap256kPlainSHA256;

// Addresses of the arrays for found passwords & success flags
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordsPlainSHA256;
__device__ __constant__ uint8_t *deviceGlobalFoundPasswordFlagsPlainSHA256;

__device__ __constant__ uint8_t *deviceGlobalStartPointsPlainSHA256;
__device__ __constant__ uint32_t *deviceGlobalStartPasswords32PlainSHA256;

__device__ __constant__ uint32_t deviceNumberStepsToRunPlainSHA256;
__device__ __constant__ uint64_t deviceNumberThreadsPlainSHA256;


// Defined if we are using the loadPasswords32/storePasswords32
#define USE_NEW_PASSWORD_LOADING 1

// Define SHA256 rotate left/right operators


__device__ inline void checkHashList256BE(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
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
        current_hash_value = DEVICE_Hashes_32[8 * search_index];
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

    while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 8])) {
        search_index--;
    }
    while ((a == DEVICE_Hashes_32[search_index * 8])) {
        if (b == DEVICE_Hashes_32[search_index * 8 + 1]) {
            if (c == DEVICE_Hashes_32[search_index * 8 + 2]) {
                if (d == DEVICE_Hashes_32[search_index * 8 + 3]) {
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
                    deviceGlobalFoundPasswordFlags[search_index] = (unsigned char) MFN_PASSWORD_SHA256;
                }
            }
        }
        search_index++;
    }
}



#define SHR(x,n) ((x & 0xFFFFFFFF) >> n)
#define ROTR(x,n) (SHR(x,n) | (x << (32 - n)))

#define S0(x) (ROTR(x, 7) ^ ROTR(x,18) ^  SHR(x, 3))
#define S1(x) (ROTR(x,17) ^ ROTR(x,19) ^  SHR(x,10))

#define S2(x) (ROTR(x, 2) ^ ROTR(x,13) ^ ROTR(x,22))
#define S3(x) (ROTR(x, 6) ^ ROTR(x,11) ^ ROTR(x,25))

#define F0(x,y,z) ((x & y) | (z & (x | y)))
#define F1(x,y,z) (z ^ (x & (y ^ z)))

#define R(t)                                    \
(                                               \
    W[t] = S1(W[t -  2]) + W[t -  7] +          \
           S0(W[t - 15]) + W[t - 16]            \
)

#define P(a,b,c,d,e,f,g,h,x,K)                  \
{                                               \
    temp1 = h + S3(e) + F1(e,f,g) + K + x;      \
    temp2 = S2(a) + F0(a,b,c);                  \
    d += temp1; h = temp1 + temp2;              \
}


#define CUDA_SHA256_FULL() { \
    uint32_t temp1, temp2; \
    a = 0x6A09E667; \
    b = 0xBB67AE85; \
    c = 0x3C6EF372; \
    d = 0xA54FF53A; \
    e = 0x510E527F; \
    f = 0x9B05688C; \
    g = 0x1F83D9AB; \
    h = 0x5BE0CD19; \
    P( a, b, c, d, e, f, g, h,  b0, 0x428A2F98 ); \
    P( h, a, b, c, d, e, f, g,  b1, 0x71374491 ); \
    P( g, h, a, b, c, d, e, f,  b2, 0xB5C0FBCF ); \
    P( f, g, h, a, b, c, d, e,  b3, 0xE9B5DBA5 ); \
    P( e, f, g, h, a, b, c, d,  b4, 0x3956C25B ); \
    P( d, e, f, g, h, a, b, c,  b5, 0x59F111F1 ); \
    P( c, d, e, f, g, h, a, b,  b6, 0x923F82A4 ); \
    P( b, c, d, e, f, g, h, a,  b7, 0xAB1C5ED5 ); \
    P( a, b, c, d, e, f, g, h,  b8, 0xD807AA98 ); \
    P( h, a, b, c, d, e, f, g,  b9, 0x12835B01 ); \
    P( g, h, a, b, c, d, e, f, b10, 0x243185BE ); \
    P( f, g, h, a, b, c, d, e, b11, 0x550C7DC3 ); \
    P( e, f, g, h, a, b, c, d, b12, 0x72BE5D74 ); \
    P( d, e, f, g, h, a, b, c, b13, 0x80DEB1FE ); \
    P( c, d, e, f, g, h, a, b, b14, 0x9BDC06A7 ); \
    P( b, c, d, e, f, g, h, a, b15, 0xC19BF174 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0xE49B69C1 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0xEFBE4786 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x0FC19DC6 ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x240CA1CC ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x2DE92C6F ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x4A7484AA ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x5CB0A9DC ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x76F988DA ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0x983E5152 ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0xA831C66D ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0xB00327C8 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0xBF597FC7 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0xC6E00BF3 ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xD5A79147 ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0x06CA6351 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0x14292967 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0x27B70A85 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0x2E1B2138 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x4D2C6DFC ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x53380D13 ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x650A7354 ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x766A0ABB ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x81C2C92E ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x92722C85 ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0xA2BFE8A1 ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0xA81A664B ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0xC24B8B70 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0xC76C51A3 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0xD192E819 ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xD6990624 ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0xF40E3585 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0x106AA070 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0x19A4C116 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0x1E376C08 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x2748774C ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x34B0BCB5 ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x391C0CB3 ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x4ED8AA4A ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x5B9CCA4F ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x682E6FF3 ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0x748F82EE ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0x78A5636F ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0x84C87814 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0x8CC70208 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0x90BEFFFA ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xA4506CEB ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0xBEF9A3F7 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0xC67178F2 ); \
    a += 0x6A09E667; \
    b += 0xBB67AE85; \
    c += 0x3C6EF372; \
    d += 0xA54FF53A; \
    e += 0x510E527F; \
    f += 0x9B05688C; \
    g += 0x1F83D9AB; \
    h += 0x5BE0CD19; \
}

// Extern storage for the plains.
extern __shared__ uint32_t plainStorageSHA256[]; \

#define MAKE_MFN_SHA256_KERNEL1_8LENGTH(pass_len) \
__global__ void MFNHashTypePlainCUDA_SHA256_GeneratedKernel_##pass_len () { \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t a, b, c, d, e, f, g, h; \
    uint32_t password_count = 0, passOffset; \
    __shared__ uint8_t __align__(16) sharedCharsetPlainSHA256[MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t __align__(16) sharedReverseCharsetPlainSHA256[MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * pass_len]; \
    __shared__ uint8_t sharedCharsetLengthsPlainSHA256[pass_len]; \
    __shared__ uint8_t __align__(16) sharedBitmap[8192]; \
    if (threadIdx.x == 0) { \
        uint64_t *sharedCharset64 = (uint64_t *)sharedCharsetPlainSHA256; \
        uint64_t *deviceCharset64 = (uint64_t *)deviceCharsetPlainSHA256; \
        uint64_t *sharedReverseCharset64 = (uint64_t *)sharedReverseCharsetPlainSHA256; \
        uint64_t *deviceReverseCharset64 = (uint64_t *)deviceReverseCharsetPlainSHA256; \
        uint64_t *constantBitmap64 = (uint64_t *)constantBitmapAPlainSHA256; \
        uint64_t *sharedBitmap64 = (uint64_t *)sharedBitmap; \
        for (a = 0; a < ((MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * pass_len) / 8); a++) { \
            sharedCharset64[a] = deviceCharset64[a]; \
            sharedReverseCharset64[a] = deviceReverseCharset64[a]; \
        } \
        for (a = 0; a < pass_len; a++) { \
            sharedCharsetLengthsPlainSHA256[a] = charsetLengthsPlainSHA256[a]; \
        } \
        for (a = 0; a < 8192 / 8; a++) { \
            sharedBitmap64[a] = constantBitmap64[a]; \
        } \
    } \
    syncthreads(); \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
    b15 = pass_len * 8; \
    loadPasswords32(deviceGlobalStartPasswords32PlainSHA256, deviceNumberThreadsPlainSHA256, pass_len); \
    while (password_count < deviceNumberStepsToRunPlainSHA256) { \
        plainStorageSHA256[threadIdx.x] = b0; \
        if (pass_len > 3) {plainStorageSHA256[threadIdx.x + blockDim.x] = b1;} \
        if (pass_len > 7) {plainStorageSHA256[threadIdx.x + 2*blockDim.x] = b2;} \
        if (pass_len > 11) {plainStorageSHA256[threadIdx.x + 3*blockDim.x] = b3;} \
        if (pass_len > 15) {plainStorageSHA256[threadIdx.x + 4*blockDim.x] = b4;} \
        b15 = pass_len * 8; \
        CUDA_SHA256_FULL(); \
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0; \
        b0 = plainStorageSHA256[threadIdx.x]; \
        if (pass_len > 3) {b1 = plainStorageSHA256[threadIdx.x + blockDim.x];} \
        if (pass_len > 7) {b2 = plainStorageSHA256[threadIdx.x + 2*blockDim.x];} \
        if (pass_len > 11) {b3 = plainStorageSHA256[threadIdx.x + 3*blockDim.x];} \
        if (pass_len > 15) {b4 = plainStorageSHA256[threadIdx.x + 4*blockDim.x];} \
            if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
                if (!(deviceGlobalBitmap256kPlainSHA256) || ((deviceGlobalBitmap256kPlainSHA256[(a >> 3) & 0x0003FFFF] >> (a & 0x7)) & 0x1)) { \
                    if (!(deviceGlobalBitmapAPlainSHA256) || ((deviceGlobalBitmapAPlainSHA256[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) { \
                        if (!deviceGlobalBitmapDPlainSHA256 || ((deviceGlobalBitmapDPlainSHA256[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) { \
                            if (!deviceGlobalBitmapCPlainSHA256 || ((deviceGlobalBitmapCPlainSHA256[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) { \
                                if (!deviceGlobalBitmapBPlainSHA256 || ((deviceGlobalBitmapBPlainSHA256[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) { \
                                    checkHashList256BE(a, b, c, d, b0, b1, b2, b3, \
                                        deviceGlobalFoundPasswordsPlainSHA256, deviceGlobalFoundPasswordFlagsPlainSHA256, \
                                        deviceGlobalHashlistAddressPlainSHA256, numberOfHashesPlainSHA256, \
                                        passwordLengthPlainSHA256); \
            }   }   }   }   }   } \
        if (charsetLengthsPlainSHA256[1] == 0) { \
                makeMFNSingleIncrementorsSHA##pass_len (sharedCharsetPlainSHA256, sharedReverseCharsetPlainSHA256, sharedCharsetLengthsPlainSHA256); \
        } else { \
                makeMFNMultipleIncrementorsSHA##pass_len (sharedCharsetPlainSHA256, sharedReverseCharsetPlainSHA256, sharedCharsetLengthsPlainSHA256); \
        } \
        password_count++; \
    } \
    storePasswords32(deviceGlobalStartPasswords32PlainSHA256, deviceNumberThreadsPlainSHA256, pass_len); \
}


MAKE_MFN_SHA256_KERNEL1_8LENGTH(1);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(2);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(3);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(4);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(5);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(6);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(7);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(8);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(9);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(10);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(11);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(12);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(13);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(14);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(15);
MAKE_MFN_SHA256_KERNEL1_8LENGTH(16);

extern "C" cudaError_t MFNHashTypePlainCUDA_SHA256_CopyValueToConstant(
        const char *symbolName, void *hostDataAddress, size_t bytesToCopy) {
    return cudaMemcpyToSymbol(symbolName, hostDataAddress, bytesToCopy);
}

extern "C" cudaError_t MFNHashTypePlainCUDA_SHA256_LaunchKernel(uint32_t passwordLength, uint32_t Blocks, uint32_t Threads) {
    //printf("MFNHashTypePlainCUDA_SHA256_LaunchKernel()\n");

    //cudaPrintfInit();
    
    // Calculate the amount of shared memory needed for the SHA256 kernels.
    // This is used to store the passwords between operations.
    int sharedMemoryBytesRequired = (((passwordLength + 1) / 4) + 1) * 4 * Threads;
    switch (passwordLength) {
        case 1:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_1 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 2:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_2 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 3:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_3 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 4:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_4 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 5:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_5 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 6:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_6 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 7:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_7 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 8:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_8 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 9:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_9 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 10:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_10 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 11:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_11 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 12:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_12 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 13:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_13 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 14:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_14 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 15:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_15 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
            break;
        case 16:
            MFNHashTypePlainCUDA_SHA256_GeneratedKernel_16 <<< Blocks, Threads, sharedMemoryBytesRequired >>> ();
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
