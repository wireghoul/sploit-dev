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
 * Crack Phpass hashes.  They are an initial round of md5($salt.$pass) for
 * length 8 salts, followed by iteration_count rounds of md5($hash.$pass) to get
 * the final hash.
 */

#define CPU_DEBUG 0

// Make my UI sane - include the files if not in the compiler environment.
#ifndef __OPENCL_VERSION__
#include "MFN_OpenCL_Common.cl"
#include "MFN_OpenCL_MD5.cl"
#include "MFN_OpenCL_PasswordCopiers.cl"
#endif

#define PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INITIAL() { \
    b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[0]); \
    if (PASSWORD_LENGTH > 3) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[1 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 7) {b4 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[2 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 11) {b5 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[3 * deviceNumberThreadsPlainPhpass]);} \
 }

#define PHPASS_STORE_PASSWORDS_TO_GLOBAL_INITIAL() { \
    vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[0]); \
    if (PASSWORD_LENGTH > 3) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[1 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 7) {vstore_type(b4, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[2 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 11) {vstore_type(b5, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[3 * deviceNumberThreadsPlainPhpass]);} \
}

#define PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INCREMENT() { \
    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[0]); \
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[1 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[2 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[3 * deviceNumberThreadsPlainPhpass]);} \
 }

#define PHPASS_STORE_PASSWORDS_TO_GLOBAL_INCREMENT() { \
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[0]); \
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[1 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[2 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[3 * deviceNumberThreadsPlainPhpass]);} \
}

#define PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INNER() { \
    b4 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[0]); \
    if (PASSWORD_LENGTH > 3) {b5 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[1 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 7) {b6 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[2 * deviceNumberThreadsPlainPhpass]);} \
    if (PASSWORD_LENGTH > 11) {b7 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainPhpass[3 * deviceNumberThreadsPlainPhpass]);} \
 }

#define PHPASS_LOAD_HASH_FROM_TEMP_SPACE() { \
    b0 = vload_type(get_global_id(0), &deviceGlobalTempSpace[0]); \
    b1 = vload_type(get_global_id(0), &deviceGlobalTempSpace[1 * deviceNumberThreadsPlainPhpass]); \
    b2 = vload_type(get_global_id(0), &deviceGlobalTempSpace[2 * deviceNumberThreadsPlainPhpass]); \
    b3 = vload_type(get_global_id(0), &deviceGlobalTempSpace[3 * deviceNumberThreadsPlainPhpass]); \
}    

#define PHPASS_STORE_HASH_TO_TEMP_SPACE() { \
    vstore_type(a, get_global_id(0), &deviceGlobalTempSpace[0]); \
    vstore_type(b, get_global_id(0), &deviceGlobalTempSpace[1 * deviceNumberThreadsPlainPhpass]); \
    vstore_type(c, get_global_id(0), &deviceGlobalTempSpace[2 * deviceNumberThreadsPlainPhpass]); \
    vstore_type(d, get_global_id(0), &deviceGlobalTempSpace[3 * deviceNumberThreadsPlainPhpass]); \
}    


// Try to organize the attributes a bit better...
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_Phpass(
    /* Bitmaps - 8/16kb, 256kb, main global bitmaps*/
    __constant unsigned char const * restrict constantBitmapAPlainPhpass, /* 0 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainPhpass, /* 1 */
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainPhpass, /* 2 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainPhpass, /* 3 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainPhpass, /* 4 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainPhpass, /* 5 */
    /* Found password data */
    __private unsigned long const numberOfHashesPlainPhpass, /* 6 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainPhpass, /* 7 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainPhpass, /* 8 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainPhpass, /* 9 */
    /* Start point data */
    __global   unsigned int * deviceGlobalStartPasswordsPlainPhpass, /* 10 */
    /* Run data - numbers, steps, etc */    
    __private unsigned long const deviceNumberThreadsPlainPhpass, /* 11 */
    __private unsigned int const deviceNumberStepsToRunPlainPhpass, /* 12 */
    /* Salt related data */
    __private unsigned long const deviceNumberOfSaltValues, /* 13 */
    __global   unsigned int const * restrict deviceGlobalSaltLengthsPhpass, /* 14 */ 
    __global   unsigned int const * restrict deviceGlobalSaltValuesPhpass, /* 15 */ 
    __private unsigned int const deviceStartingSaltOffsetPhpass, /* 16 */
    /* Iteration start point data */
    __global  unsigned int const * restrict deviceGlobalIterationCounts, /* 17 */
    __private unsigned int const deviceIterationStartCount, /* 18 */
    __global  unsigned int * deviceGlobalTempSpace /* 19 */
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    
    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    
    unsigned int saltIndex;
    unsigned int saltLength;
    unsigned int iterations;

#if CPU_DEBUG
    printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Number hashes: %d\n", numberOfHashesPlainPhpass);
        printf("Bitmap A: %lu\n", deviceGlobalBitmapAPlainPhpass);
        printf("Bitmap B: %lu\n", deviceGlobalBitmapBPlainPhpass);
        printf("Bitmap C: %lu\n", deviceGlobalBitmapCPlainPhpass);
        printf("Bitmap D: %lu\n", deviceGlobalBitmapDPlainPhpass);
        printf("Number threads: %lu\n", deviceNumberThreadsPlainPhpass);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainPhpass);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        printf("Number salts: %u\n", deviceNumberOfSaltValues);
        printf("Starting salt offset: %u\n", deviceStartingSaltOffsetPhpass);
        printf("Starting iteration offset: %u\n", deviceIterationStartCount);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainPhpass[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    saltIndex = deviceStartingSaltOffsetPhpass;
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    //printf("Salt index: %d\n", saltIndex);

    if (deviceIterationStartCount == 0) {
        //printf("Start iteration 0: Will load pass/salt right now.\n");
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesPhpass[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesPhpass[(1 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        //printf("Salt: %c%c%c%c%c%c%c%c\n",
        //        (b0) & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff,
        //        (b1) & 0xff, (b1 >> 8) & 0xff, (b1 >> 16) & 0xff, (b1 >> 24) & 0xff);
        PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INITIAL();
        b14 = (PASSWORD_LENGTH + 8) * 8;
        MD5_FULL_HASH();
        //printf("Initial result %08x %08x %08x %08x\n", a, b, c, d);
        b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        b0 = a; b1 = b; b2 = c; b3 = d;
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
        //printf("Loaded hash from temp: %08x %08x %08x %08x\n", b0, b1, b2, b3);
    }

    // In either case, load passwords to the inner loop count and set length.
    b14 = (PASSWORD_LENGTH + 16) * 8;
    PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INNER();

    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        //printf("Step %d\n", password_count);
        //printf("iterations remaining: %u\n", iterations);
        
        MD5_FULL_HASH();
        b0 = a; b1 = b; b2 = c; b3 = d;
        //printf ("Hash: %08x %08x %08x %08x\n", a, b, c, d);
        iterations--;
        
        if (!iterations) {
            //printf("Iterations reached 0, doing check and reloading.\n");
            OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                //printf("Resetting salt index to 0 from %d\n", saltIndex);
                saltIndex = 0;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INCREMENT();
                //printf("Pre-increment Password: %c%c%c%c\n", b0 & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff);
                /*printf("%d Pre-increment Password .s0: %c%c%c%c\n", get_global_id(0),
                        (b0.s0) & 0xff, (b0.s0 >> 8) & 0xff, (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff);
                printf("%d Pre-increment Password .s1: %c%c%c%c\n", get_global_id(0),
                        (b0.s1) & 0xff, (b0.s1 >> 8) & 0xff, (b0.s1 >> 16) & 0xff, (b0.s1 >> 24) & 0xff);
                printf("%d Pre-increment Password .s2: %c%c%c%c\n", get_global_id(0),
                        (b0.s2) & 0xff, (b0.s2 >> 8) & 0xff, (b0.s2 >> 16) & 0xff, (b0.s2 >> 24) & 0xff);
                printf("%d Pre-increment Password .s3: %c%c%c%c\n", get_global_id(0),
                        (b0.s3) & 0xff, (b0.s3 >> 8) & 0xff, (b0.s3 >> 16) & 0xff, (b0.s3 >> 24) & 0xff);*/

                OpenCLNoMemPasswordIncrementorLE();

                //printf("Post-increment Password: %c%c%c%c\n", b0 & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff);
                /*printf("%d Post-increment Password .s0: %c%c%c%c\n", get_global_id(0),
                        (b0.s0) & 0xff, (b0.s0 >> 8) & 0xff, (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff);
                printf("%d Post-increment Password .s1: %c%c%c%c\n", get_global_id(0),
                        (b0.s1) & 0xff, (b0.s1 >> 8) & 0xff, (b0.s1 >> 16) & 0xff, (b0.s1 >> 24) & 0xff);
                printf("%d Post-increment Password .s2: %c%c%c%c\n", get_global_id(0),
                        (b0.s2) & 0xff, (b0.s2 >> 8) & 0xff, (b0.s2 >> 16) & 0xff, (b0.s2 >> 24) & 0xff);
                printf("%d Post-increment Password .s3: %c%c%c%c\n", get_global_id(0),
                        (b0.s3) & 0xff, (b0.s3 >> 8) & 0xff, (b0.s3 >> 16) & 0xff, (b0.s3 >> 24) & 0xff);*/

                PHPASS_STORE_PASSWORDS_TO_GLOBAL_INCREMENT();
            }
            //printf("New salt index: %d\n", saltIndex);
            iterations = deviceGlobalIterationCounts[saltIndex];
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            if (get_local_id(0) == 0) {
                saltPrefetch[0] = deviceGlobalSaltValuesPhpass[(0 * deviceNumberOfSaltValues) + saltIndex];
                saltPrefetch[1] = deviceGlobalSaltValuesPhpass[(1 * deviceNumberOfSaltValues) + saltIndex];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            b0 = saltPrefetch[0];
            b1 = saltPrefetch[1];
            //printf("Salt: %c%c%c%c%c%c%c%c\n",
            //        (b0) & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff,
            //        (b1) & 0xff, (b1 >> 8) & 0xff, (b1 >> 16) & 0xff, (b1 >> 24) & 0xff);
            PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INITIAL();
            b14 = (PASSWORD_LENGTH + 8) * 8;
            MD5_FULL_HASH();
            //printf("Initial result %08x %08x %08x %08x\n", a, b, c, d);
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
            b14 = (PASSWORD_LENGTH + 16) * 8;
            PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INNER();
        }

        password_count++;
    }
    //printf("Storing hash to temp: %08x %08x %08x %08x\n", a, b, c, d);
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}
