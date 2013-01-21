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
#endif


// For 1-4 input blocks (passwords length 0-15)
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_MD5WL_B1_4(
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
    __private unsigned long const deviceNumberThreads, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5, /* 16 */
    __global   unsigned char const * restrict deviceWordlistLengths, /* 17 */
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 18 */
    __private unsigned int const deviceNumberWords, /* 19 */
    __private unsigned int const deviceStartStep, /* 20 */
    __private unsigned char const deviceNumberBlocksPerWord /* 21 */
        
) {
    // Start the kernel.
    //__local unsigned char sharedCharsetPlainMD5[MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedReverseCharsetPlainMD5[MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedCharsetLengthsPlainMD5[PASSWORD_LENGTH];
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

#if CPU_DEBUG 
    printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Number hashes: %d\n", numberOfHashesPlainMD5);
        printf("Number threads: %lu\n", deviceNumberThreads);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainMD5);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        printf("Start Step: %d\n", deviceStartStep);
        printf("Blocks per word: %d\n", deviceNumberBlocksPerWord);
        printf("Number words: %d\n", deviceNumberWords);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.


        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreads);
#endif        
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
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
            deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
            deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
            deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
            numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);

        password_count++;
        passwordStep++;
    }

}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_MD5WL_B5_8(
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
    __private unsigned long const deviceNumberThreads, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5, /* 16 */
    __global   unsigned char const * restrict deviceWordlistLengths, /* 17 */
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 18 */
    __private unsigned int const deviceNumberWords, /* 19 */
    __private unsigned int const deviceStartStep, /* 20 */
    __private unsigned char const deviceNumberBlocksPerWord /* 21 */
        
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

#if CPU_DEBUG 
    printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Number hashes: %d\n", numberOfHashesPlainMD5);
        printf("Number threads: %lu\n", deviceNumberThreads);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainMD5);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        printf("Start Step: %d\n", deviceStartStep);
        printf("Blocks per word: %d\n", deviceNumberBlocksPerWord);
        printf("Number words: %d\n", deviceNumberWords);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.


        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreads);
#endif        
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
        // Load length.
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
        
        MD5_FULL_HASH();
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
            deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
            deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
            deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
            numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);

        password_count++;
        passwordStep++;
    }
}

__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_MD5WL_B9_14(
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
    __private unsigned long const deviceNumberThreads, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5, /* 16 */
    __global   unsigned char const * restrict deviceWordlistLengths, /* 17 */
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 18 */
    __private unsigned int const deviceNumberWords, /* 19 */
    __private unsigned int const deviceStartStep, /* 20 */
    __private unsigned char const deviceNumberBlocksPerWord /* 21 */
        
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.


        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[7 * deviceNumberWords + passwordStep * deviceNumberThreads]);
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

        // Load length.
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));
        
        MD5_FULL_HASH();
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
            deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
            deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
            deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
            numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);

        password_count++;
        passwordStep++;
    }
}



// Kernel for block length 15-16 - data goes in, but the actual hash happens
// in two blocks, with the second block being zero except for the length.
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_MD5WL_B15_16(
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
    __private unsigned long const deviceNumberThreads, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5, /* 16 */
    __global   unsigned char const * restrict deviceWordlistLengths, /* 17 */
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 18 */
    __private unsigned int const deviceNumberWords, /* 19 */
    __private unsigned int const deviceStartStep, /* 20 */
    __private unsigned char const deviceNumberBlocksPerWord /* 21 */
        
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.


        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
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

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        // Load length.
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));

        MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
            deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
            deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
            deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
            numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);

        password_count++;
        passwordStep++;
    }
}


// Kernel for block length 17-30
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_MD5WL_B17_30(
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
    __private unsigned long const deviceNumberThreads, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5, /* 16 */
    __global   unsigned char const * restrict deviceWordlistLengths, /* 17 */
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 18 */
    __private unsigned int const deviceNumberWords, /* 19 */
    __private unsigned int const deviceStartStep, /* 20 */
    __private unsigned char const deviceNumberBlocksPerWord /* 21 */
        
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.


        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
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
        // Load moar data!
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
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
            deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
            deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
            deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
            numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);

        password_count++;
        passwordStep++;
    }
}


// Kernel for block length 17-30
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_MD5WL_B31_32(
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
    __private unsigned long const deviceNumberThreads, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5, /* 16 */
    __global   unsigned char const * restrict deviceWordlistLengths, /* 17 */
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 18 */
    __private unsigned int const deviceNumberWords, /* 19 */
    __private unsigned int const deviceStartStep, /* 20 */
    __private unsigned char const deviceNumberBlocksPerWord /* 21 */
        
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.


        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
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
        // Load moar data!
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

        // Load length.
        b14 = 8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads]));

        MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);

        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
            deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
            deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
            deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
            numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);

        password_count++;
        passwordStep++;
    }
}
