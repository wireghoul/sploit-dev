/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2012  Bitweasil (http://www.cryptohaze.com/)

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
 * This kernel implements IPB cracking.  The algorithm is a salted algorithm:
 * md5(md5($salt).md5($pass)) - salt is loaded as 32 ASCII bytes, already pre-
 * hashed - no need to do this multiple times.
 */

//#define CPU_DEBUG 0

// Make my UI sane - include the files if not in the compiler environment.
#ifndef __OPENCL_VERSION__
#include "MFN_OpenCL_Common.cl"
#include "MFN_OpenCL_MD5.cl"
#include "MFN_OpenCL_PasswordCopiers.cl"
#include "MFN_OpenCL_BIN2HEX.cl"
#endif

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_IPB_MAX_CHARSET_LENGTH 128


/**
 * Define the kernel arguments for all the kernels.  They are always going to be
 * identical, so there is no point in duplicating them needlessly.
 */
#define __OPENCL_IPBWL_KERNEL_ARGS__ \
    __constant unsigned char const * restrict constantBitmapAPlainIPB, /* 0 */ \
\
    __private unsigned long const numberOfHashesPlainIPB, /* 1 */ \
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainIPB, /* 2 */ \
    __global   unsigned char *deviceGlobalFoundPasswordsPlainIPB, /* 3 */ \
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainIPB, /* 4 */ \
\
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainIPB, /* 5 */ \
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainIPB, /* 6 */ \
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainIPB, /* 7 */ \
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainIPB, /* 8 */ \
\
    __private unsigned long const deviceNumberThreads, /* 9 */ \
    __private unsigned int const deviceNumberStepsToRunPlainIPB, /* 10 */ \
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainIPB, /* 11 */ \
    __global   unsigned int const * restrict deviceGlobalSaltLengthsIPB, /* 12 */  \
    __global   unsigned int const * restrict deviceGlobalSaltValuesIPB, /* 13 */  \
    __private unsigned long const deviceNumberOfSaltValues, /* 14 */ \
    __private unsigned int const deviceStartingSaltOffsetIPB, /* 15 */ \
    __global   unsigned char const * restrict deviceWordlistLengths, /* 16 */ \
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 17 */ \
    __private unsigned int const deviceNumberWords, /* 18 */ \
    __private unsigned int const deviceStartStep, /* 19 */ \
    __private unsigned char const deviceNumberBlocksPerWord /* 20 */


/**
 * Expand a password's hash in a-d to ASCII in b8-b15.
 */
#define IPB_EXPAND_PASSWORD_TO_ASCII() { \
    b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0; \
    AddHashCharacterAsString_LE_LE(hashLookup, a, b8, bitmap_index); \
    AddHashCharacterAsString_LE_LE(hashLookup, a, b9, bitmap_index); \
    AddHashCharacterAsString_LE_LE(hashLookup, b, b10, bitmap_index); \
    AddHashCharacterAsString_LE_LE(hashLookup, b, b11, bitmap_index); \
    AddHashCharacterAsString_LE_LE(hashLookup, c, b12, bitmap_index); \
    AddHashCharacterAsString_LE_LE(hashLookup, c, b13, bitmap_index); \
    AddHashCharacterAsString_LE_LE(hashLookup, d, b14, bitmap_index); \
    AddHashCharacterAsString_LE_LE(hashLookup, d, b15, bitmap_index); \
}

/*
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_IPBWL_Prototype(
    __OPENCL_IPBWL_KERNEL_ARGS__
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    // Temp space to load the salt (32 hex characters, 8 words).
    __local unsigned int saltPrefetch[8];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    vector_type salt_a, salt_b, salt_c, salt_d;
    
    unsigned long password_count = 0;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type lookupResult;
    
    unsigned int saltIndex;

    uint passwordStep = deviceStartStep;

#if CPU_DEBUG
    printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Number hashes: %d\n", numberOfHashesPlainIPB);
        printf("Bitmap A: %lu\n", deviceGlobalBitmapAPlainIPB);
        printf("Bitmap B: %lu\n", deviceGlobalBitmapBPlainIPB);
        printf("Bitmap C: %lu\n", deviceGlobalBitmapCPlainIPB);
        printf("Bitmap D: %lu\n", deviceGlobalBitmapDPlainIPB);
        printf("Number threads: %lu\n", deviceNumberThreads);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainIPB);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        printf("Number salts: %u\n", deviceNumberOfSaltValues);
        printf("Starting salt offset: %u\n", deviceStartingSaltOffsetIPB);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainIPB[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][1] = hexLookupValues[counter % 16];
            hashLookup[counter][0] = hexLookupValues[counter / 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    saltIndex = deviceStartingSaltOffsetIPB;
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
    if (deviceNumberBlocksPerWord >= 2) {
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
    }
    if (deviceNumberBlocksPerWord >= 3) {
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
    }
    if (deviceNumberBlocksPerWord >= 4) {
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
    }
    // Load length.
    b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
    
    MD5_FULL_HASH();
    // a, b, c, d contain the password hash.  Expand it into b8-b15
    b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    AddHashCharacterAsString_LE_LE(hashLookup, a, b8, bitmap_index);
    AddHashCharacterAsString_LE_LE(hashLookup, a, b9, bitmap_index);
    AddHashCharacterAsString_LE_LE(hashLookup, b, b10, bitmap_index);
    AddHashCharacterAsString_LE_LE(hashLookup, b, b11, bitmap_index);
    AddHashCharacterAsString_LE_LE(hashLookup, c, b12, bitmap_index);
    AddHashCharacterAsString_LE_LE(hashLookup, c, b13, bitmap_index);
    AddHashCharacterAsString_LE_LE(hashLookup, d, b14, bitmap_index);
    AddHashCharacterAsString_LE_LE(hashLookup, d, b15, bitmap_index);
    
    while (password_count < deviceNumberStepsToRunPlainIPB) {
        //printf("Step %d\n", password_count);
        //printf("saltIndex: %u\n", saltIndex);
        //printf("saltLength: %u\n", saltLength);
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesIPB[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesIPB[(1 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[2] = deviceGlobalSaltValuesIPB[(2 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[3] = deviceGlobalSaltValuesIPB[(3 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[4] = deviceGlobalSaltValuesIPB[(4 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[5] = deviceGlobalSaltValuesIPB[(5 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[6] = deviceGlobalSaltValuesIPB[(6 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[7] = deviceGlobalSaltValuesIPB[(7 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //b0 = vload_type(0, &deviceGlobalSaltValuesIPB[0 + saltIndex]);
        //b1 = vload_type(0, &deviceGlobalSaltValuesIPB[deviceNumberOfSaltValues + saltIndex]);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        
        //printf("Salt .s0: %c%c%c%c%c\n", (b0.s0) & 0xff, (b0.s0 >> 8) & 0xff, (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff, b1.s0 & 0xff);
        //printf("Salt .s1: %c%c%c%c%c\n", (b0.s1) & 0xff, (b0.s1 >> 8) & 0xff, (b0.s1 >> 16) & 0xff, (b0.s1 >> 24) & 0xff, b1.s1 & 0xff);
        //printf("b0.s0: %08x\n", b0.s0);
        //printf("b1.s0: %08x\n", b1.s0);
        //printf("b0.s1: %08x\n", b0.s1);
        //printf("b1.s1: %08x\n", b1.s1);
        // Do the salt hash - get a vector of results.
        MD5_FULL_HASH_LEN5();
        
        //printf("Salt hash a.s0: %08x\n", a.s0);
        //printf("Salt hash a.s1: %08x\n", a.s1);
        
        salt_a = a; 
        salt_b = b;
        salt_c = c;
        salt_d = d;

#if grt_vector_1
        // Do the non-vector version of this.
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = (vector_type)0;
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a, b0);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a, b1);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b, b2);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b, b3);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c, b4);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c, b5);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d, b6);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d, b7);
        MD5_FULL_HASH();
        prev_a = a;
        prev_b = b;
        prev_c = c;
        prev_d = d;
        MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d);
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
            deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
            deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
            deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
            numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
#elif grt_vector_2
        IPB_HASH_SALT_STEP(0);
        IPB_HASH_SALT_STEP(1);
#elif grt_vector_4
        IPB_HASH_SALT_STEP(0);
        IPB_HASH_SALT_STEP(1);
        IPB_HASH_SALT_STEP(2);
        IPB_HASH_SALT_STEP(3);
#elif grt_vector_8
        IPB_HASH_SALT_STEP(0);
        IPB_HASH_SALT_STEP(1);
        IPB_HASH_SALT_STEP(2);
        IPB_HASH_SALT_STEP(3);
        IPB_HASH_SALT_STEP(4);
        IPB_HASH_SALT_STEP(5);
        IPB_HASH_SALT_STEP(6);
        IPB_HASH_SALT_STEP(7);
#elif grt_vector_16
        IPB_HASH_SALT_STEP(0);
        IPB_HASH_SALT_STEP(1);
        IPB_HASH_SALT_STEP(2);
        IPB_HASH_SALT_STEP(3);
        IPB_HASH_SALT_STEP(4);
        IPB_HASH_SALT_STEP(5);
        IPB_HASH_SALT_STEP(6);
        IPB_HASH_SALT_STEP(7);
        IPB_HASH_SALT_STEP(8);
        IPB_HASH_SALT_STEP(9);
        IPB_HASH_SALT_STEP(A);
        IPB_HASH_SALT_STEP(B);
        IPB_HASH_SALT_STEP(C);
        IPB_HASH_SALT_STEP(D);
        IPB_HASH_SALT_STEP(E);
        IPB_HASH_SALT_STEP(F);
#endif
        
        //        //printf(".s0 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s0 >> 0) & 0xff, (b0.s0 >> 8) & 0xff,
//                (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff,
//                (b1.s0 >> 0) & 0xff,
//                a.s0, b.s0, c.s0, d.s0);
//        //printf(".s1 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s1 >> 0) & 0xff, (b0.s1 >> 8) & 0xff,
//                (b0.s1 >> 16) & 0xff, (b0.s1 >> 24) & 0xff,
//                (b1.s1 >> 0) & 0xff,
//                a.s1, b.s1, c.s1, d.s1);
//        //printf(".s2 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s2 >> 0) & 0xff, (b0.s2 >> 8) & 0xff,
//                (b0.s2 >> 16) & 0xff, (b0.s2 >> 24) & 0xff,
//                (b1.s2 >> 0) & 0xff,
//                a.s2, b.s2, c.s2, d.s2);
//        //printf(".s3 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s3 >> 0) & 0xff, (b0.s3 >> 8) & 0xff,
//                (b0.s3 >> 16) & 0xff, (b0.s3 >> 24) & 0xff,
//                (b1.s3 >> 0) & 0xff,
//                a.s3, b.s3, c.s3, d.s3);
  
        saltIndex+=VECTOR_WIDTH;
        if (saltIndex >= deviceNumberOfSaltValues) {
            //printf("Resetting salt index to 0 from %d\n", saltIndex);
            saltIndex = 0;
            passwordStep++;
            
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            if (deviceNumberBlocksPerWord >= 2) {
                b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            }
            if (deviceNumberBlocksPerWord >= 3) {
                b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            }
            if (deviceNumberBlocksPerWord >= 4) {
                b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            }
            // Load length.
            b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));

            MD5_FULL_HASH();
            //printf("Raw password hash .s0: %08x %08x %08x %08x\n", a.s0, b.s0, c.s0, d.s0);
            //printf("Raw password hash .s1: %08x %08x %08x %08x\n", a.s1, b.s1, c.s1, d.s1);

            // a, b, c, d contain the password hash.  Expand it into b8-b15
            b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            AddHashCharacterAsString_LE_LE(hashLookup, a, b8, bitmap_index);
            AddHashCharacterAsString_LE_LE(hashLookup, a, b9, bitmap_index);
            AddHashCharacterAsString_LE_LE(hashLookup, b, b10, bitmap_index);
            AddHashCharacterAsString_LE_LE(hashLookup, b, b11, bitmap_index);
            AddHashCharacterAsString_LE_LE(hashLookup, c, b12, bitmap_index);
            AddHashCharacterAsString_LE_LE(hashLookup, c, b13, bitmap_index);
            AddHashCharacterAsString_LE_LE(hashLookup, d, b14, bitmap_index);
            AddHashCharacterAsString_LE_LE(hashLookup, d, b15, bitmap_index);
        }
        //printf("Incrementing password_count by %d\n", ((saltIndex + VECTOR_WIDTH) < deviceNumberOfSaltValues) ? VECTOR_WIDTH : (deviceNumberOfSaltValues - saltIndex));
        password_count+= ((saltIndex + VECTOR_WIDTH) < deviceNumberOfSaltValues) ? VECTOR_WIDTH : (deviceNumberOfSaltValues - saltIndex);
    }
    // Do NOT store the b0, b1, etc - it's salt data, not the password!
}
*/

__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_IPBWL_B1_4(
    __OPENCL_IPBWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    __local unsigned int saltPrefetch[8];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    unsigned long password_count = 0;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type lookupResult;
    unsigned int saltIndex;
    uint passwordStep = deviceStartStep;
    
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainIPB[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][1] = hexLookupValues[counter % 16];
            hashLookup[counter][0] = hexLookupValues[counter / 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    saltIndex = deviceStartingSaltOffsetIPB;

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        if (deviceNumberBlocksPerWord >= 2) {
            b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 3) {
            b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 4) {
            b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
        MD5_FULL_HASH();
        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPB) {
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesIPB[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesIPB[(1 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[2] = deviceGlobalSaltValuesIPB[(2 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[3] = deviceGlobalSaltValuesIPB[(3 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[4] = deviceGlobalSaltValuesIPB[(4 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[5] = deviceGlobalSaltValuesIPB[(5 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[6] = deviceGlobalSaltValuesIPB[(6 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[7] = deviceGlobalSaltValuesIPB[(7 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        b2 = saltPrefetch[2];
        b3 = saltPrefetch[3];
        b4 = saltPrefetch[4];
        b5 = saltPrefetch[5];
        b6 = saltPrefetch[6];
        b7 = saltPrefetch[7];
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainIPB, \
                deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, \
                deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB,  \
                deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB, \
                numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB); \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValues) {
            saltIndex = 0;
            passwordStep++;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                if (deviceNumberBlocksPerWord >= 2) {
                    b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 3) {
                    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 4) {
                    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
                MD5_FULL_HASH();
                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    }
}



__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_IPBWL_B5_14(
    __OPENCL_IPBWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    __local unsigned int saltPrefetch[8];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    unsigned long password_count = 0;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type lookupResult;
    unsigned int saltIndex;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainIPB[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][1] = hexLookupValues[counter % 16];
            hashLookup[counter][0] = hexLookupValues[counter / 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    saltIndex = deviceStartingSaltOffsetIPB;
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        if (deviceNumberBlocksPerWord >= 5) {
            b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 6) {
            b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 7) {
            b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 8) {
            b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 9) {
            b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 10) {
            b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 11) {
            b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 12) {
            b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 13) {
            b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 14) {
            b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
        MD5_FULL_HASH();
        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPB) {
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesIPB[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesIPB[(1 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[2] = deviceGlobalSaltValuesIPB[(2 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[3] = deviceGlobalSaltValuesIPB[(3 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[4] = deviceGlobalSaltValuesIPB[(4 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[5] = deviceGlobalSaltValuesIPB[(5 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[6] = deviceGlobalSaltValuesIPB[(6 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[7] = deviceGlobalSaltValuesIPB[(7 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        b2 = saltPrefetch[2];
        b3 = saltPrefetch[3];
        b4 = saltPrefetch[4];
        b5 = saltPrefetch[5];
        b6 = saltPrefetch[6];
        b7 = saltPrefetch[7];
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainIPB, \
                deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, \
                deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB,  \
                deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB, \
                numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB); \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValues) {
            //printf("Resetting salt index to 0 from %d\n", saltIndex);
            saltIndex = 0;
            passwordStep++;
            
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                if (deviceNumberBlocksPerWord >= 5) {
                    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 6) {
                    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 7) {
                    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 8) {
                    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 9) {
                    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 10) {
                    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 11) {
                    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 12) {
                    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 13) {
                    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 14) {
                    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
                MD5_FULL_HASH();
                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_IPBWL_B15_16(
    __OPENCL_IPBWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    __local unsigned int saltPrefetch[8];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    unsigned long password_count = 0;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type lookupResult;
    unsigned int saltIndex;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainIPB[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][1] = hexLookupValues[counter % 16];
            hashLookup[counter][0] = hexLookupValues[counter / 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    saltIndex = deviceStartingSaltOffsetIPB;
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        if (deviceNumberBlocksPerWord >= 15) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[14 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 16) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[15 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        MD5_FULL_HASH();
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
        MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);

        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPB) {
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesIPB[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesIPB[(1 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[2] = deviceGlobalSaltValuesIPB[(2 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[3] = deviceGlobalSaltValuesIPB[(3 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[4] = deviceGlobalSaltValuesIPB[(4 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[5] = deviceGlobalSaltValuesIPB[(5 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[6] = deviceGlobalSaltValuesIPB[(6 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[7] = deviceGlobalSaltValuesIPB[(7 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        b2 = saltPrefetch[2];
        b3 = saltPrefetch[3];
        b4 = saltPrefetch[4];
        b5 = saltPrefetch[5];
        b6 = saltPrefetch[6];
        b7 = saltPrefetch[7];
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainIPB, \
                deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, \
                deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB,  \
                deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB, \
                numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB); \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValues) {
            //printf("Resetting salt index to 0 from %d\n", saltIndex);
            saltIndex = 0;
            passwordStep++;
            
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                if (deviceNumberBlocksPerWord >= 15) {
                    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[14 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 16) {
                    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[15 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
                MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    }
}



__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_IPBWL_B17_30(
    __OPENCL_IPBWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    __local unsigned int saltPrefetch[8];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    unsigned long password_count = 0;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type lookupResult;
    unsigned int saltIndex;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainIPB[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][1] = hexLookupValues[counter % 16];
            hashLookup[counter][0] = hexLookupValues[counter / 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    saltIndex = deviceStartingSaltOffsetIPB;
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[14 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[15 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        MD5_FULL_HASH();
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        if (deviceNumberBlocksPerWord >= 17) {
            b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[16 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 18) {
            b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[17 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 19) {
            b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[18 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 20) {
            b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[19 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 21) {
            b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[20 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 22) {
            b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[21 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 23) {
            b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[22 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 24) {
            b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[23 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 25) {
            b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[24 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 26) {
            b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[25 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 27) {
            b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[26 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 28) {
            b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[27 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 29) {
            b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[28 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 30) {
            b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[29 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }

        // Load length.
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));

        MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPB) {
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesIPB[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesIPB[(1 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[2] = deviceGlobalSaltValuesIPB[(2 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[3] = deviceGlobalSaltValuesIPB[(3 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[4] = deviceGlobalSaltValuesIPB[(4 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[5] = deviceGlobalSaltValuesIPB[(5 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[6] = deviceGlobalSaltValuesIPB[(6 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[7] = deviceGlobalSaltValuesIPB[(7 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        b2 = saltPrefetch[2];
        b3 = saltPrefetch[3];
        b4 = saltPrefetch[4];
        b5 = saltPrefetch[5];
        b6 = saltPrefetch[6];
        b7 = saltPrefetch[7];
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainIPB, \
                deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, \
                deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB,  \
                deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB, \
                numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB); \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValues) {
            //printf("Resetting salt index to 0 from %d\n", saltIndex);
            saltIndex = 0;
            passwordStep++;
            
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[14 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[15 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                if (deviceNumberBlocksPerWord >= 17) {
                    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[16 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 18) {
                    b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[17 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 19) {
                    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[18 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 20) {
                    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[19 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 21) {
                    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[20 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 22) {
                    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[21 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 23) {
                    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[22 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 24) {
                    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[23 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 25) {
                    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[24 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 26) {
                    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[25 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 27) {
                    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[26 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 28) {
                    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[27 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 29) {
                    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[28 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 30) {
                    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[29 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_IPBWL_B31_32(
    __OPENCL_IPBWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    __local unsigned int saltPrefetch[8];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    unsigned long password_count = 0;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type lookupResult;
    unsigned int saltIndex;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainIPB[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][1] = hexLookupValues[counter % 16];
            hashLookup[counter][0] = hexLookupValues[counter / 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    saltIndex = deviceStartingSaltOffsetIPB;

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[14 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[15 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        MD5_FULL_HASH();
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[16 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[17 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[18 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[19 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[20 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[21 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[22 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[23 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[24 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[25 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[26 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[27 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[28 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[29 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        if (deviceNumberBlocksPerWord >= 31) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[30 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 32) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[31 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
        MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);

        IPB_EXPAND_PASSWORD_TO_ASCII();
    }
    while (password_count < deviceNumberStepsToRunPlainIPB) {
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesIPB[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesIPB[(1 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[2] = deviceGlobalSaltValuesIPB[(2 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[3] = deviceGlobalSaltValuesIPB[(3 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[4] = deviceGlobalSaltValuesIPB[(4 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[5] = deviceGlobalSaltValuesIPB[(5 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[6] = deviceGlobalSaltValuesIPB[(6 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[7] = deviceGlobalSaltValuesIPB[(7 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        b2 = saltPrefetch[2];
        b3 = saltPrefetch[3];
        b4 = saltPrefetch[4];
        b5 = saltPrefetch[5];
        b6 = saltPrefetch[6];
        b7 = saltPrefetch[7];
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH(); \
            prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
            MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainIPB, \
                deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, \
                deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB,  \
                deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB, \
                numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB); \
        }
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValues) {
            //printf("Resetting salt index to 0 from %d\n", saltIndex);
            saltIndex = 0;
            passwordStep++;
            
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[14 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[15 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[16 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[17 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[18 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[19 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[20 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[21 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[22 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[23 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[24 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[25 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[26 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[27 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[28 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[29 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                if (deviceNumberBlocksPerWord >= 31) {
                    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[30 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 32) {
                    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[31 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
                MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);

                IPB_EXPAND_PASSWORD_TO_ASCII();
            }
        }
        password_count++;
    }
}
