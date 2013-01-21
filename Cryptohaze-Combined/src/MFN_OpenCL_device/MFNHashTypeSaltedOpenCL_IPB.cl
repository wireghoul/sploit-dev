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
 * md5(md5($salt).md5($pass)) - salt is 5 characters.
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


#define IPB_LOAD_PASSWORDS_FROM_GLOBAL() { \
    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[0]); \
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[1 * deviceNumberThreadsPlainIPB]);} \
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[2 * deviceNumberThreadsPlainIPB]);} \
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[3 * deviceNumberThreadsPlainIPB]);} \
 }

#define IPB_STORE_PASSWORDS_TO_GLOBAL() { \
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[0]); \
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[1 * deviceNumberThreadsPlainIPB]);} \
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[2 * deviceNumberThreadsPlainIPB]);} \
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainIPB[3 * deviceNumberThreadsPlainIPB]);} \
}

// Perform the step for one vector suffix.
#define IPB_HASH_SALT_STEP(suffix) { \
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = (vector_type)0; \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a.s##suffix, b0); \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a.s##suffix, b1); \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b.s##suffix, b2); \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b.s##suffix, b3); \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c.s##suffix, b4); \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c.s##suffix, b5); \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d.s##suffix, b6); \
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d.s##suffix, b7); \
        MD5_FULL_HASH(); \
        prev_a = a; prev_b = b; prev_c = c; prev_d = d; \
        MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d); \
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, \
            deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, \
            deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB,  \
            deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB, \
            numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB); \
}

__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_IPB(
    __constant unsigned char const * restrict deviceCharsetPlainIPB, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainIPB, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainIPB, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainIPB, /* 3 */
        
    __private unsigned long const numberOfHashesPlainIPB, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainIPB, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainIPB, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainIPB, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainIPB, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainIPB, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainIPB, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainIPB, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainIPB, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainIPB, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainIPB, /* 14 */
    __global   unsigned int * deviceGlobalStartPasswordsPlainIPB, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainIPB, /* 16 */
    __global   unsigned int const * restrict deviceGlobalSaltLengthsIPB, /* 17 */ 
    __global   unsigned int const * restrict deviceGlobalSaltValuesIPB, /* 18 */ 
    __private unsigned long const deviceNumberOfSaltValues, /* 19 */
    __private unsigned int const deviceStartingSaltOffsetIPB /* 20 */
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type b0_t, b1_t, b2_t, b3_t;
    vector_type prev_a, prev_b, prev_c, prev_d;
    vector_type salt_a, salt_b, salt_c, salt_d;
    
    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    
    unsigned int saltIndex;
    unsigned int saltLength;

#if CPU_DEBUG || 1
    //printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        //printf("Number hashes: %d\n", numberOfHashesPlainIPB);
        //printf("Bitmap A: %lu\n", deviceGlobalBitmapAPlainIPB);
        //printf("Bitmap B: %lu\n", deviceGlobalBitmapBPlainIPB);
        //printf("Bitmap C: %lu\n", deviceGlobalBitmapCPlainIPB);
        //printf("Bitmap D: %lu\n", deviceGlobalBitmapDPlainIPB);
        //printf("Number threads: %lu\n", deviceNumberThreadsPlainIPB);
        //printf("Steps to run: %u\n", deviceNumberStepsToRunPlainIPB);
        //printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        //printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        //printf("Number salts: %u\n", deviceNumberOfSaltValues);
        //printf("Starting salt offset: %u\n", deviceStartingSaltOffsetIPB);
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
    
    // Load the initial password from global memory.
    IPB_LOAD_PASSWORDS_FROM_GLOBAL();

    // Hash the password.
    b14 = PASSWORD_LENGTH * 8;
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
        saltLength = deviceGlobalSaltLengthsIPB[saltIndex];
        //printf("saltLength: %u\n", saltLength);
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = vload_type(0, &deviceGlobalSaltValuesIPB[0 + saltIndex]);
            saltPrefetch[1] = vload_type(0, &deviceGlobalSaltValuesIPB[deviceNumberOfSaltValues + saltIndex]);
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
        /*
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = (vector_type)0;
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a.s0, b0);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a.s0, b1);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b.s0, b2);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b.s0, b3);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c.s0, b4);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c.s0, b5);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d.s0, b6);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d.s0, b7);
        MD5_FULL_HASH();
        prev_a = a;
        prev_b = b;
        prev_c = c;
        prev_d = d;
        MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d);
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
            deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
            deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
            deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
            numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
        
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = (vector_type)0;
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a.s1, b0);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_a.s1, b1);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b.s1, b2);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_b.s1, b3);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c.s1, b4);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_c.s1, b5);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d.s1, b6);
        AddHashCharacterAsString_LE_LE_VE(hashLookup, salt_d.s1, b7);
        
        //printf("b0.s0: %08x %c%c%c%c\n", b0.s0, (b0.s0) & 0xff, (b0.s0 >> 8) & 0xff, (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff);
        //printf("b1.s0: %08x %c%c%c%c\n", b1.s0, (b1.s0) & 0xff, (b1.s0 >> 8) & 0xff, (b1.s0 >> 16) & 0xff, (b1.s0 >> 24) & 0xff);
        
        MD5_FULL_HASH();
        prev_a = a;
        prev_b = b;
        prev_c = c;
        prev_d = d;
        MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d);
        
        //printf("Final a: %08x\n", a.s0);
        
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
            deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
            deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
            deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
            numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
        */
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
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
            deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
            deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
            deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
            numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
#elif grt_vector_2
        IPB_HASH_SALT_STEP(0);
        IPB_HASH_SALT_STEP(1);
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
                    deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
                    deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
                    deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
                    numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
#elif grt_vector_4
        IPB_HASH_SALT_STEP(0);
        IPB_HASH_SALT_STEP(1);
        IPB_HASH_SALT_STEP(2);
        IPB_HASH_SALT_STEP(3);
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
                    deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
                    deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
                    deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
                    numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
#elif grt_vector_8
        IPB_HASH_SALT_STEP(0);
        IPB_HASH_SALT_STEP(1);
        IPB_HASH_SALT_STEP(2);
        IPB_HASH_SALT_STEP(3);
        IPB_HASH_SALT_STEP(4);
        IPB_HASH_SALT_STEP(5);
        IPB_HASH_SALT_STEP(6);
        IPB_HASH_SALT_STEP(7);
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
                    deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
                    deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
                    deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
                    numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
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
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainIPB, 
                    deviceGlobalBitmapBPlainIPB, deviceGlobalBitmapCPlainIPB, 
                    deviceGlobalBitmapDPlainIPB, deviceGlobalHashlistAddressPlainIPB, 
                    deviceGlobalFoundPasswordsPlainIPB, deviceGlobalFoundPasswordFlagsPlainIPB,
                    numberOfHashesPlainIPB, deviceGlobal256kbBitmapAPlainIPB);
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
  
        saltIndex++;
        if (saltIndex >= deviceNumberOfSaltValues) {
            //printf("Resetting salt index to 0 from %d\n", saltIndex);
            saltIndex = 0;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            IPB_LOAD_PASSWORDS_FROM_GLOBAL();
            //printf("Pre-increment Password: %c%c%c%c\n", b0 & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff);
            //printf("Pre-increment Password .s0: %c%c\n", b0.s0 & 0xff, (b0.s0 >> 8) & 0xff);
            //printf("Pre-increment Password .s1: %c%c\n", b0.s1 & 0xff, (b0.s1 >> 8) & 0xff);

            OpenCLNoMemPasswordIncrementorLE();
            
            //printf("Post-increment Password: %c%c%c%c\n", b0 & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff);
            //printf("Post-increment Password .s0: %c%c\n", b0.s0 & 0xff, (b0.s0 >> 8) & 0xff);
            //printf("Post-increment Password .s1: %c%c\n", b0.s1 & 0xff, (b0.s1 >> 8) & 0xff);
            IPB_STORE_PASSWORDS_TO_GLOBAL();
            
            b14 = PASSWORD_LENGTH * 8;
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
