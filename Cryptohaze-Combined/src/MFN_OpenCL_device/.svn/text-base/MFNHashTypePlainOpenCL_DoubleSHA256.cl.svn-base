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
 * Implements SHA256 brute forcing for lengths 0-55 (one block length).  Done
 * in the proper fashion with sane includes & such!
 */

#define CPU_DEBUG 1

// Make my UI sane - include the files if not in the compiler environment.
#ifndef __OPENCL_VERSION__
#include "MFN_OpenCL_Common.cl"
#include "MFN_OpenCL_SHA256.cl"
#include "MFN_OpenCL_BIN2HEX.cl"
#include "MFN_OpenCL_PasswordCopiers.cl"
#endif

#define print_hash(num, vector) printf("%d: %08x %08x %08x %08x %08x\n", num, a.vector, b.vector, c.vector, d.vector, e.vector);

#define print_all_hash(num) { \
print_hash(num, s0); \
}

#define reverse(x)(x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24);

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_DOUBLESHA256_MAX_CHARSET_LENGTH 128

__kernel    
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_DoubleSHA256(
    __constant unsigned char const * restrict deviceCharsetPlainDoubleSHA256, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainDoubleSHA256, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainDoubleSHA256, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainDoubleSHA256, /* 3 */
        
    __private unsigned long const numberOfHashesPlainDoubleSHA256, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainDoubleSHA256, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainDoubleSHA256, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainDoubleSHA256, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainDoubleSHA256, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainDoubleSHA256, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainDoubleSHA256, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainDoubleSHA256, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainDoubleSHA256, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainDoubleSHA256, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainDoubleSHA256, /* 14 */
    __global   unsigned int * deviceGlobalStartPasswordsPlainDoubleSHA256, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainDoubleSHA256 /* 16 */
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    //__local unsigned int  plainStore[(((PASSWORD_LENGTH + 1)/4) + 1) * THREADSPERBLOCK * VECTOR_WIDTH];
     __local unsigned char hashLookup[256][2];
   
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    vector_type a, b, c, d, e, f, g, h, bitmap_index;
    // Not checking other than a,b,c,d - don't need the others saved.
    vector_type prev_a, prev_b, prev_c, prev_d;
    vector_type b0_t, b1_t, b2_t, b3_t;
    
    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainDoubleSHA256[counter];
        }
 #pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][1] = hexLookupValues[counter % 16];
            hashLookup[counter][0] = hexLookupValues[counter / 16];
        }
   }
    barrier(CLK_LOCAL_MEM_FENCE);

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    b15 = (vector_type) (PASSWORD_LENGTH * 8);
    a = b = c = d = e = f = g = h = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[0]);
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[1 * deviceNumberThreadsPlainDoubleSHA256]);}
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[2 * deviceNumberThreadsPlainDoubleSHA256]);}
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[3 * deviceNumberThreadsPlainDoubleSHA256]);}
    
    while (password_count < deviceNumberStepsToRunPlainDoubleSHA256) {
        // Store the plains - they get mangled by SHA256 state.
//        vstore_type(b0, get_local_id(0), &plainStore[0]);
//        if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_local_id(0), &plainStore[VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_local_id(0), &plainStore[2 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_local_id(0),&plainStore[3 * VECTOR_WIDTH * THREADSPERBLOCK]);}

            
        b0_t = b0;
        if (PASSWORD_LENGTH > 3) {b1_t = b1;} 
        if (PASSWORD_LENGTH > 7) {b2_t = b2;}
        if (PASSWORD_LENGTH > 11) {b3_t = b3;}        
        
        b15 = (vector_type) (PASSWORD_LENGTH * 8);
        OPENCL_SHA256_FULL_CONSTANTS();
        
        //printf("plain: %c%c%c%c\n", (b0_t >> 24) & 0xff, (b0_t >> 16) & 0xff,
        //        (b0_t >> 8) & 0xff, b0_t & 0xff);
        //printf("hash: %08x %08x %08x %08x\n      %08x %08x %08x %08x\n",
        //        a, b, c, d, e, f, g, h);
        
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        
        AddHashCharacterAsString_BE_BE(hashLookup, a, b0, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, a, b1, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, b, b2, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, b, b3, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, c, b4, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, c, b5, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, d, b6, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, d, b7, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, e, b8, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, e, b9, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, f, b10, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, f, b11, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, g, b12, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, g, b13, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, h, b14, bitmap_index);
        AddHashCharacterAsString_BE_BE(hashLookup, h, b15, bitmap_index);
        
        //printf("Post-converting:\n");
        //printf("b0: %08x\n", b0);
        //printf("b1: %08x\n", b1);
        OPENCL_SHA256_FULL_CONSTANTS();
        prev_a = a;
        prev_b = b;
        prev_c = c;
        prev_d = d;

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        b0 = 0x80000000;
        b15 = (vector_type) (64 * 8);

        OPENCL_SHA256_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d, 0, 0, 0, 0);
        //printf("Second round done\n");
        //printf("hash: %08x %08x %08x %08x\n      %08x %08x %08x %08x\n",
        //        a, b, c, d, e, f, g, h);
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        
        b0 = b0_t;
        if (PASSWORD_LENGTH > 3) {b1 = b1_t;} 
        if (PASSWORD_LENGTH > 7) {b2 = b2_t;}
        if (PASSWORD_LENGTH > 11) {b3 = b3_t;}        

//        b0 = vload_type(get_local_id(0), &plainStore[0]);
//        if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_local_id(0), &plainStore[VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_local_id(0), &plainStore[2 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_local_id(0),&plainStore[3 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        
//        printf(".s0 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s0 >> 24) & 0xff, (b0.s0 >> 16) & 0xff,
//                (b0.s0 >> 8) & 0xff, (b0.s0 >> 0) & 0xff,
//                (b1.s0 >> 24) & 0xff,
//                a.s0, b.s0, c.s0, d.s0, e.s0);
//        printf(".s1 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s1 >> 24) & 0xff, (b0.s1 >> 16) & 0xff,
//                (b0.s1 >> 8) & 0xff, (b0.s1 >> 0) & 0xff,
//                (b1.s1 >> 24) & 0xff,
//                a.s1, b.s1, c.s1, d.s1, e.s1);
//        printf(".s2 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s2 >> 24) & 0xff, (b0.s2 >> 16) & 0xff,
//                (b0.s2 >> 8) & 0xff, (b0.s2 >> 0) & 0xff,
//                (b1.s2 >> 24) & 0xff,
//                a.s2, b.s2, c.s2, d.s2, e.s2);
//        printf(".s3 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s3 >> 24) & 0xff, (b0.s3 >> 16) & 0xff,
//                (b0.s3 >> 8) & 0xff, (b0.s3 >> 0) & 0xff,
//                (b1.s3 >> 24) & 0xff,
//                a.s3, b.s3, c.s3, d.s3, e.s3);
//        

        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainDoubleSHA256, 
            deviceGlobalBitmapBPlainDoubleSHA256, deviceGlobalBitmapCPlainDoubleSHA256, 
            deviceGlobalBitmapDPlainDoubleSHA256, deviceGlobalHashlistAddressPlainDoubleSHA256, 
            deviceGlobalFoundPasswordsPlainDoubleSHA256, deviceGlobalFoundPasswordFlagsPlainDoubleSHA256,
            numberOfHashesPlainDoubleSHA256, deviceGlobal256kbBitmapAPlainDoubleSHA256);

        OpenCLNoMemPasswordIncrementorBE();

        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[0]);
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[1 * deviceNumberThreadsPlainDoubleSHA256]);}
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[2 * deviceNumberThreadsPlainDoubleSHA256]);}
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainDoubleSHA256[3 * deviceNumberThreadsPlainDoubleSHA256]);}
  
}
