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
 * Implements the MD5 wordlist cracking for lengths 0-127.
 */

//#define CPU_DEBUG 0

// Make my UI sane - include the files if not in the compiler environment.
#ifndef __OPENCL_VERSION__
#include "MFN_OpenCL_Common.cl"
#include "MFN_OpenCL_MD5.cl"
#include "MFN_OpenCL_PasswordCopiers.cl"
#include "MFN_OpenCL_BIN2HEX.cl"
#endif

__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_DoubleMD5(
    __constant unsigned char const * restrict deviceCharsetPlainMD5, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainMD5, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainMD5, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainMD5, /* 3 */
        
    __private unsigned long const numberOfHashesPlainMD5, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainMD5, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainMD5, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainMD5, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainMD5, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainMD5, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainMD5, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainMD5, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainMD5, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainMD5, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5 /* 16 */
) {
    // Start the kernel.
    //__local unsigned char sharedCharsetPlainMD5[MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedReverseCharsetPlainMD5[MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedCharsetLengthsPlainMD5[PASSWORD_LENGTH];
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char hashLookup[256][2];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type b0_t, b1_t, b2_t, b3_t, b4_t, b5_t, b6_t, b7_t;

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
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }
        for (counter = 0; counter < 256; counter++) {
            hashLookup[counter][0] = hexLookupValues[counter / 16];
            hashLookup[counter][1] = hexLookupValues[counter % 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    b14 = (vector_type) (PASSWORD_LENGTH * 8);
    a = b = c = d = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[0]);
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[1 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[2 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[3 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 15) {b4 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[4 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 19) {b5 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[5 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 23) {b6 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[6 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 27) {b7 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[7 * deviceNumberThreadsPlainMD5]);}
        
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        MD5_FULL_HASH();
        b0_t = b0;
        b1_t = b1;
        b2_t = b2;
        b3_t = b3;
        b4_t = b4;
        b5_t = b5;
        b6_t = b6;
        b7_t = b7;

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        
        AddHashCharacterAsString_LE_LE(hashLookup, a, b0, bitmap_index);
        AddHashCharacterAsString_LE_LE(hashLookup, a, b1, bitmap_index);
        AddHashCharacterAsString_LE_LE(hashLookup, b, b2, bitmap_index);
        AddHashCharacterAsString_LE_LE(hashLookup, b, b3, bitmap_index);
        AddHashCharacterAsString_LE_LE(hashLookup, c, b4, bitmap_index);
        AddHashCharacterAsString_LE_LE(hashLookup, c, b5, bitmap_index);
        AddHashCharacterAsString_LE_LE(hashLookup, d, b6, bitmap_index);
        AddHashCharacterAsString_LE_LE(hashLookup, d, b7, bitmap_index);
        
        MD5_FULL_HASH_32_ASCII();

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        b0 = b0_t;
        b1 = b1_t;
        b2 = b2_t;
        b3 = b3_t;
        b4 = b4_t;
        b5 = b5_t;
        b6 = b6_t;
        b7 = b7_t;
        
        b14 = (vector_type) (PASSWORD_LENGTH * 8);

        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
            deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
            deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
            deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
            numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);
        
        OpenCLNoMemPasswordIncrementorLE();
        
        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[0]);
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[1 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[2 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[3 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 15) {vstore_type(b4, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[4 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 19) {vstore_type(b5, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[5 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 23) {vstore_type(b6, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[6 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 27) {vstore_type(b7, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[7 * deviceNumberThreadsPlainMD5]);}
}
