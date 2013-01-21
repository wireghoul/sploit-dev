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
 * the final hash.  This is the wordlist kernel and is, so far, the most complex
 * kernel I've written!
 */


#define CPU_DEBUG 0

#if CPU_DEBUG
#define phpass_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define phpass_printf(fmt, ...) do {} while (0)
#endif

// Make my UI sane - include the files if not in the compiler environment.
#ifndef __OPENCL_VERSION__
#include "MFN_OpenCL_Common.cl"
#include "MFN_OpenCL_MD5.cl"
#include "MFN_OpenCL_PasswordCopiers.cl"
#endif


#define PHPASS_LOAD_HASH_FROM_TEMP_SPACE() { \
    b0 = vload_type(get_global_id(0), &deviceGlobalTempSpace[0]); \
    b1 = vload_type(get_global_id(0), &deviceGlobalTempSpace[1 * deviceNumberThreads]); \
    b2 = vload_type(get_global_id(0), &deviceGlobalTempSpace[2 * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceGlobalTempSpace[3 * deviceNumberThreads]); \
}    

#define PHPASS_STORE_HASH_TO_TEMP_SPACE() { \
    vstore_type(a, get_global_id(0), &deviceGlobalTempSpace[0]); \
    vstore_type(b, get_global_id(0), &deviceGlobalTempSpace[1 * deviceNumberThreads]); \
    vstore_type(c, get_global_id(0), &deviceGlobalTempSpace[2 * deviceNumberThreads]); \
    vstore_type(d, get_global_id(0), &deviceGlobalTempSpace[3 * deviceNumberThreads]); \
}    

// It's not that I'm OCD... I'm CDO.  With the letters alphabetized properly.
#define __OPENCL_PHPASSWL_KERNEL_ARGS__ \
             /* Bitmaps - 8/16kb, 256kb, main global bitmaps*/ \
    /*  0 */ __constant unsigned char const * restrict constantBitmapAPlainPhpass, \
    /*  1 */ __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainPhpass, \
    /*  2 */ __global   unsigned char const * restrict deviceGlobalBitmapAPlainPhpass, \
    /*  3 */ __global   unsigned char const * restrict deviceGlobalBitmapBPlainPhpass, \
    /*  4 */ __global   unsigned char const * restrict deviceGlobalBitmapCPlainPhpass, \
    /*  5 */ __global   unsigned char const * restrict deviceGlobalBitmapDPlainPhpass, \
             /* Found password data */ \
    /*  6 */ __private  unsigned long const numberOfHashesPlainPhpass, \
    /*  7 */ __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainPhpass, \
    /*  8 */ __global   unsigned char *deviceGlobalFoundPasswordsPlainPhpass, \
    /*  9 */ __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainPhpass, \
             /* Run data - numbers, steps, etc */ \
    /* 10 */ __private  unsigned long const deviceNumberThreads, \
    /* 11 */ __private  unsigned int const deviceNumberStepsToRunPlainPhpass, \
             /* Salt related data */ \
    /* 12 */ __private  unsigned long const deviceNumberOfSaltValues, \
    /* 13 */ __global   unsigned int const * restrict deviceGlobalSaltLengthsPhpass, \
    /* 14 */ __global   unsigned int const * restrict deviceGlobalSaltValuesPhpass, \
    /* 15 */ __private  unsigned int const deviceStartingSaltOffsetPhpass, \
             /* Iteration related data */ \
    /* 16 */ __global   unsigned int const * restrict deviceGlobalIterationCounts, \
    /* 17 */ __private  unsigned int const deviceIterationStartCount, \
    /* 18 */ __global   unsigned int * deviceGlobalTempSpace, \
             /* Wordlist data */ \
    /* 19 */ __global   unsigned char const * restrict deviceWordlistLengths, \
    /* 20 */ __global   unsigned int const * restrict deviceWordlistBlocks, \
    /* 21 */ __private  unsigned int const deviceNumberWords, \
    /* 22 */ __private  unsigned char const deviceNumberBlocksPerWord, \
    /* 23 */ __private  unsigned int const deviceStartWord \


/*
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B1_4(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
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
    uint passwordStep = deviceStartWord;

#if CPU_DEBUG
    phpass_printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        phpass_printf("Number hashes: %d\n", numberOfHashesPlainPhpass);
        phpass_printf("Number threads: %lu\n", deviceNumberThreads);
        phpass_printf("Steps to run: %u\n", deviceNumberStepsToRunPlainPhpass);
        phpass_printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        phpass_printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        phpass_printf("Number salts: %u\n", deviceNumberOfSaltValues);
        phpass_printf("Starting salt offset: %u\n", deviceStartingSaltOffsetPhpass);
        phpass_printf("Starting iteration offset: %u\n", deviceIterationStartCount);
        phpass_printf("Number words: %u\n", deviceNumberWords);
        phpass_printf("Start word: %u\n", deviceStartWord);
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
    phpass_printf("Salt index: %d\n", saltIndex);

    if (deviceIterationStartCount == 0) {
        phpass_printf("Start iteration 0: Will load pass/salt right now.\n");
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        if (get_local_id(0) == 0) {
            saltPrefetch[0] = deviceGlobalSaltValuesPhpass[(0 * deviceNumberOfSaltValues) + saltIndex];
            saltPrefetch[1] = deviceGlobalSaltValuesPhpass[(1 * deviceNumberOfSaltValues) + saltIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        b0 = saltPrefetch[0];
        b1 = saltPrefetch[1];
        phpass_printf("Salt: %c%c%c%c%c%c%c%c\n",
                (b0) & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff,
                (b1) & 0xff, (b1 >> 8) & 0xff, (b1 >> 16) & 0xff, (b1 >> 24) & 0xff);
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            if (deviceNumberBlocksPerWord >= 2) {
                b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            }
            if (deviceNumberBlocksPerWord >= 3) {
                b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            }
            if (deviceNumberBlocksPerWord >= 4) {
                b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
            }
            b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
            MD5_FULL_HASH();
            phpass_printf("Initial result %08x %08x %08x %08x\n", a, b, c, d);
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
        phpass_printf("Loaded hash from temp: %08x %08x %08x %08x\n", b0, b1, b2, b3);
    }

    // In either case, load passwords to the inner loop count and set length.
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        if (deviceNumberBlocksPerWord >= 2) {
            b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 3) {
            b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        if (deviceNumberBlocksPerWord >= 4) {
            b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
        }
        b14 = 128 + (8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
    }
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        phpass_printf("Step %d\n", password_count);
        phpass_printf("iterations remaining: %u\n", iterations);

        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH();
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
        //printf ("Hash: %08x %08x %08x %08x\n", a, b, c, d);
        iterations--;
        
        if (!iterations) {
            phpass_printf("Iterations reached 0, doing check and reloading.\n");
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                phpass_printf("Resetting salt index to 0 from %d\n", saltIndex);
                saltIndex = 0;
                passwordStep++;
            }
            phpass_printf("New salt index: %d\n", saltIndex);
            iterations = deviceGlobalIterationCounts[saltIndex];
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            if (get_local_id(0) == 0) {
                saltPrefetch[0] = deviceGlobalSaltValuesPhpass[(0 * deviceNumberOfSaltValues) + saltIndex];
                saltPrefetch[1] = deviceGlobalSaltValuesPhpass[(1 * deviceNumberOfSaltValues) + saltIndex];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            b0 = saltPrefetch[0];
            b1 = saltPrefetch[1];
            phpass_printf("Salt: %c%c%c%c%c%c%c%c\n",
                    (b0) & 0xff, (b0 >> 8) & 0xff, (b0 >> 16) & 0xff, (b0 >> 24) & 0xff,
                    (b1) & 0xff, (b1 >> 8) & 0xff, (b1 >> 16) & 0xff, (b1 >> 24) & 0xff);
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                if (deviceNumberBlocksPerWord >= 2) {
                    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 3) {
                    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 4) {
                    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
                MD5_FULL_HASH();
                phpass_printf("Initial result %08x %08x %08x %08x\n", a, b, c, d);
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
                b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                if (deviceNumberBlocksPerWord >= 2) {
                    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 3) {
                    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                if (deviceNumberBlocksPerWord >= 4) {
                    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreads]);
                }
                b14 = 128 + (8 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
            }
        }

        password_count++;
    }
    phpass_printf("Storing hash to temp: %08x %08x %08x %08x\n", a, b, c, d);
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}
*/

/**
 * Clear all the word blocks and load the salt into the proper positions.
 * 
 * This will load the salt into the temp region for the block, then expand that
 * into the registers for each thread on all cores.
 */
#define PHPASS_LOAD_SALT() { \
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = \
        b14 = b15 = (vector_type)0; \
    if (get_local_id(0) == 0) { \
        saltPrefetch[0] = deviceGlobalSaltValuesPhpass[ \
            (0 * deviceNumberOfSaltValues) + saltIndex]; \
        saltPrefetch[1] = deviceGlobalSaltValuesPhpass[ \
            (1 * deviceNumberOfSaltValues) + saltIndex]; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    b0 = saltPrefetch[0]; \
    b1 = saltPrefetch[1]; \
}

/**
 * Password loaders into the two positions.  The OUTER functions load the
 * password block starting at b2, whereas the INNER functions load the password
 * starting at b4.  The final number set is the block count it handles.
 * 
 * These require deviceNumberThreads, deviceNumberWords, passwordStep, and
 * deviceWordlistBlocks to be set.
 */
#define PHPASS_LOAD_PASSWORD_OUTER_1_4() { \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 2) { \
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 3) { \
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 4) { \
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}

#define PHPASS_LOAD_PASSWORD_INNER_1_4() { \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 2) { \
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 3) { \
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 4) { \
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 128 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}

#define PHPASS_LOAD_PASSWORD_OUTER_5_12() { \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        4 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 6) { \
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            5 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 7) { \
        b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            6 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 8) { \
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            7 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 9) { \
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            8 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 10) { \
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            9 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 11) { \
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            10 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 12) { \
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            11 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}

#define PHPASS_LOAD_PASSWORD_INNER_5_10() { \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        4 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 6) { \
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            5 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 7) { \
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            6 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 8) { \
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            7 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 9) { \
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            8 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 10) { \
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            9 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 128 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}

#define PHPASS_LOAD_PASSWORD_INNER_11_12() { \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        4 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        5 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        6 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        7 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        8 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        9 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        10 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 12) { \
        b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            11 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
}

#define PHPASS_LOAD_PASSWORD_OUTER_13_14() { \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        4 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        5 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        6 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        7 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        8 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        9 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        10 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        11 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        12 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 14) { \
        b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            13 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
}

/**
 * Block loaders for the first block.  Used when multiple iterations are needed.
 */
#define PHPASS_LOAD_PASSWORD_INNER_FIRST_BLOCK() { \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        4 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        5 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        6 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        7 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        8 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        9 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        10 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        11 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
}

#define PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK() { \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        0 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        1 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        2 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        3 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        4 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        5 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        6 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        7 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        8 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        9 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        10 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        11 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        12 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        13 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
}

#define PHPASS_LOAD_PASSWORD_INNER_13_26() { \
    if (deviceNumberBlocksPerWord >= 13) { \
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            12 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 14) { \
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            13 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 15) { \
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            14 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 16) { \
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            15 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 17) { \
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            16 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 18) { \
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            17 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 19) { \
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            18 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 20) { \
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            19 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 21) { \
        b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            20 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 22) { \
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            21 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 23) { \
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            22 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 24) { \
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            23 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 25) { \
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            24 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 26) { \
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            25 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 128 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}

#define PHPASS_LOAD_PASSWORD_OUTER_15_28() { \
    if (deviceNumberBlocksPerWord >= 15) { \
        b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            14 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 16) { \
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            15 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 17) { \
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            16 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 18) { \
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            17 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 19) { \
        b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            18 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 20) { \
        b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            19 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 21) { \
        b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            20 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 22) { \
        b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            21 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 23) { \
        b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            22 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 24) { \
        b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            23 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 25) { \
        b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            24 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 26) { \
        b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            25 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 27) { \
        b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            26 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 28) { \
        b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            27 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}

#define PHPASS_LOAD_PASSWORD_INNER_27_28() { \
    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        12 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        13 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        14 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        15 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        16 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        17 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        18 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        19 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        20 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        21 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        22 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        23 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        24 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        25 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        26 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 28) { \
        b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            27 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
}

#define PHPASS_LOAD_PASSWORD_INNER_SECOND_BLOCK() { \
    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        12 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        13 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        14 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        15 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        16 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        17 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        18 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        19 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        20 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        21 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        22 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        23 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        24 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        25 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        26 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        27 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
}

#define PHPASS_LOAD_PASSWORD_OUTER_29_30() { \
    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        14 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        15 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        16 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        17 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        18 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        19 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        20 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        21 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        22 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        23 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        24 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        25 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        26 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        27 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        28 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 30) { \
        b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            29 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
}

#define PHPASS_LOAD_PASSWORD_OUTER_SECOND_BLOCK() { \
    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        14 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        15 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        16 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        17 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b4 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        18 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b5 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        19 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b6 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        20 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b7 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        21 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b8 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        22 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b9 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        23 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b10 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        24 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b11 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        25 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b12 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        26 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b13 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        27 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        28 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        29 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
}


#define PHPASS_LOAD_PASSWORD_INNER_29_32() { \
    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        28 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 30) { \
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            29 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 31) { \
        b2 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            30 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    if (deviceNumberBlocksPerWord >= 32) { \
        b3 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            31 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 128 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}


#define PHPASS_LOAD_PASSWORD_OUTER_31_32() { \
    b0 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
        30 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    if (deviceNumberBlocksPerWord >= 32) { \
        b1 = vload_type(get_global_id(0), &deviceWordlistBlocks[ \
            31 * deviceNumberWords + passwordStep * deviceNumberThreads]); \
    } \
    b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
}

__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B1_4(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_1_4();
            MD5_FULL_HASH();
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }

    // In either case, load passwords to the inner loop count and set length.
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        PHPASS_LOAD_PASSWORD_INNER_1_4();
    }
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH();
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_1_4();
                MD5_FULL_HASH();
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
                PHPASS_LOAD_PASSWORD_INNER_1_4();
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B5_10(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_5_12();
            MD5_FULL_HASH();
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }

    // In either case, load passwords to the inner loop count and set length.
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        PHPASS_LOAD_PASSWORD_INNER_5_10();
    }
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH();
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_5_12();
                MD5_FULL_HASH();
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
                PHPASS_LOAD_PASSWORD_INNER_5_10();
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B11_12(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13,
            b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d, pass_b14;
    
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_5_12();
            MD5_FULL_HASH();
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }

    // In either case, load passwords to the inner loop count and set length.
    if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
        PHPASS_LOAD_PASSWORD_INNER_11_12();
    }
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            pass_b14 = b14;
            b14 = 128 + (8 * convert_type(vload_type(get_global_id(0),
                &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
            MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
            b14 = pass_b14;
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_5_12();
                MD5_FULL_HASH();
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
                PHPASS_LOAD_PASSWORD_INNER_11_12();
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}

__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B13_14(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13,
            b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_13_14();
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
                &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
            MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
            // This needs to be stored for repeated loading.
            //PHPASS_STORE_HASH_TO_TEMP_SPACE();
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }
    //printf("Exiting first stage, b0... %08x %08x %08x %08x\n", b0, b1, b2, b3);
    // In either case, load passwords to the inner loop count and set length.
    // Passwords are loaded each loop.
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            // First block: Clear data, load hash, load pass into b4-b15
            // b0-b3 are loaded.
            //PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
            PHPASS_LOAD_PASSWORD_INNER_FIRST_BLOCK();
            //printf("First block data 1: %08x %08x %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5, b6, b7);
            //printf("First block data 2: %08x %08x %08x %08x %08x %08x %08x %08x\n", b8, b9, b10, b11, b12, b13, b14, b15);
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            // Load the second block.
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_INNER_13_26();
            b14 = 128 + (8 * convert_type(vload_type(get_global_id(0),
                &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
            //printf("Second block data 1: %08x %08x %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5, b6, b7);
            //printf("Second block data 2: %08x %08x %08x %08x %08x %08x %08x %08x\n", b8, b9, b10, b11, b12, b13, b14, b15);
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            b0 = a; b1 = b; b2 = c; b3 = d;
            //printf("Post data: %08x %08x %08x %08x\n", a, b, c, d);
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_13_14();
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
                    &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
                MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
                // This needs to be stored for repeated loading.
                //PHPASS_STORE_HASH_TO_TEMP_SPACE();
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B15_26(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13,
            b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_OUTER_15_28();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
            // This needs to be stored for repeated loading.
            //PHPASS_STORE_HASH_TO_TEMP_SPACE();
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }
    //printf("Exiting first stage, b0... %08x %08x %08x %08x\n", b0, b1, b2, b3);
    // In either case, load passwords to the inner loop count and set length.
    // Passwords are loaded each loop.
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            // First block: Clear data, load hash, load pass into b4-b15
            // b0-b3 are loaded.
            //PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
            PHPASS_LOAD_PASSWORD_INNER_FIRST_BLOCK();
            //printf("First block data 1: %08x %08x %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5, b6, b7);
            //printf("First block data 2: %08x %08x %08x %08x %08x %08x %08x %08x\n", b8, b9, b10, b11, b12, b13, b14, b15);
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            // Load the second block.
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_INNER_13_26();
            b14 = 128 + (8 * convert_type(vload_type(get_global_id(0),
                &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
            //printf("Second block data 1: %08x %08x %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5, b6, b7);
            //printf("Second block data 2: %08x %08x %08x %08x %08x %08x %08x %08x\n", b8, b9, b10, b11, b12, b13, b14, b15);
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            b0 = a; b1 = b; b2 = c; b3 = d;
            //printf("Post data: %08x %08x %08x %08x\n", a, b, c, d);
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                PHPASS_LOAD_PASSWORD_OUTER_15_28();
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
                // This needs to be stored for repeated loading.
                //PHPASS_STORE_HASH_TO_TEMP_SPACE();
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B27_28(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13,
            b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_OUTER_15_28();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
            // This needs to be stored for repeated loading.
            //PHPASS_STORE_HASH_TO_TEMP_SPACE();
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }
    //printf("Exiting first stage, b0... %08x %08x %08x %08x\n", b0, b1, b2, b3);
    // In either case, load passwords to the inner loop count and set length.
    // Passwords are loaded each loop.
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            // First block: Clear data, load hash, load pass into b4-b15
            // b0-b3 are loaded.
            //PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
            PHPASS_LOAD_PASSWORD_INNER_FIRST_BLOCK();
            //printf("First block data 1: %08x %08x %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5, b6, b7);
            //printf("First block data 2: %08x %08x %08x %08x %08x %08x %08x %08x\n", b8, b9, b10, b11, b12, b13, b14, b15);
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            // Load the second block.
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_INNER_27_28();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b14 = 128 + (8 * convert_type(vload_type(get_global_id(0),
                &deviceWordlistLengths[passwordStep * deviceNumberThreads])));
            MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
            b0 = a; b1 = b; b2 = c; b3 = d;
            //printf("Post data: %08x %08x %08x %08x\n", a, b, c, d);
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                PHPASS_LOAD_PASSWORD_OUTER_15_28();
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
                // This needs to be stored for repeated loading.
                //PHPASS_STORE_HASH_TO_TEMP_SPACE();
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B29_30(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13,
            b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_OUTER_29_30();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
                &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
            MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }
    //printf("Exiting first stage, b0... %08x %08x %08x %08x\n", b0, b1, b2, b3);
    // In either case, load passwords to the inner loop count and set length.
    // Passwords are loaded each loop.
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            // First block: Clear data, load hash, load pass into b4-b15
            // b0-b3 are loaded.
            //PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
            PHPASS_LOAD_PASSWORD_INNER_FIRST_BLOCK();
            //printf("First block data 1: %08x %08x %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5, b6, b7);
            //printf("First block data 2: %08x %08x %08x %08x %08x %08x %08x %08x\n", b8, b9, b10, b11, b12, b13, b14, b15);
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            // Load the second block.
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_INNER_SECOND_BLOCK();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_INNER_29_32();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            b0 = a; b1 = b; b2 = c; b3 = d;
            //printf("Post data: %08x %08x %08x %08x\n", a, b, c, d);
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                PHPASS_LOAD_PASSWORD_OUTER_29_30();
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b14 = 64 + (8 * convert_type(vload_type(get_global_id(0), \
                    &deviceWordlistLengths[passwordStep * deviceNumberThreads]))); \
                MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypeSaltedOpenCL_PhpassWL_B31_32(
    __OPENCL_PHPASSWL_KERNEL_ARGS__
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local vector_type saltPrefetch[2];

    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13,
            b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;
    
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
    uint passwordStep = deviceStartWord;

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
    if (deviceIterationStartCount == 0) {
        PHPASS_LOAD_SALT();
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_OUTER_SECOND_BLOCK();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_OUTER_31_32();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            b0 = a; b1 = b; b2 = c; b3 = d;
        }
    } else {
        PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
    }
    //printf("Exiting first stage, b0... %08x %08x %08x %08x\n", b0, b1, b2, b3);
    // In either case, load passwords to the inner loop count and set length.
    // Passwords are loaded each loop.
    iterations = deviceGlobalIterationCounts[saltIndex] - deviceIterationStartCount;

    while (password_count < deviceNumberStepsToRunPlainPhpass) {
        if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
            // First block: Clear data, load hash, load pass into b4-b15
            // b0-b3 are loaded.
            //PHPASS_LOAD_HASH_FROM_TEMP_SPACE();
            PHPASS_LOAD_PASSWORD_INNER_FIRST_BLOCK();
            //printf("First block data 1: %08x %08x %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5, b6, b7);
            //printf("First block data 2: %08x %08x %08x %08x %08x %08x %08x %08x\n", b8, b9, b10, b11, b12, b13, b14, b15);
            MD5_FULL_HASH();
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            // Load the second block.
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_INNER_SECOND_BLOCK();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            prev_a = a; prev_b = b; prev_c = c; prev_d = d;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
            PHPASS_LOAD_PASSWORD_INNER_29_32();
            MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
            b0 = a; b1 = b; b2 = c; b3 = d;
            //printf("Post data: %08x %08x %08x %08x\n", a, b, c, d);
        }
        iterations--;
        
        if (!iterations) {
            OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainPhpass, 
                    deviceGlobalBitmapBPlainPhpass, deviceGlobalBitmapCPlainPhpass, 
                    deviceGlobalBitmapDPlainPhpass, deviceGlobalHashlistAddressPlainPhpass, 
                    deviceGlobalFoundPasswordsPlainPhpass, deviceGlobalFoundPasswordFlagsPlainPhpass,
                    numberOfHashesPlainPhpass, deviceGlobal256kbBitmapAPlainPhpass);
            saltIndex++;
            if (saltIndex >= deviceNumberOfSaltValues) {
                saltIndex = 0;
                passwordStep++;
            }
            iterations = deviceGlobalIterationCounts[saltIndex];
            PHPASS_LOAD_SALT();
            if ((get_global_id(0) + (passwordStep * deviceNumberThreads)) < deviceNumberWords) {
                PHPASS_LOAD_PASSWORD_OUTER_FIRST_BLOCK();
                MD5_FULL_HASH();
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                PHPASS_LOAD_PASSWORD_OUTER_SECOND_BLOCK();
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                prev_a = a; prev_b = b; prev_c = c; prev_d = d;
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                PHPASS_LOAD_PASSWORD_OUTER_31_32();
                MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
                b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
                b0 = a; b1 = b; b2 = c; b3 = d;
            }
        }
        password_count++;
    }
    PHPASS_STORE_HASH_TO_TEMP_SPACE();
}
